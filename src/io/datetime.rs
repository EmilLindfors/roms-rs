//! Shared date/time parsing for the `io` readers.
//!
//! The NorKyst text and parquet readers and the tide-gauge reader all convert
//! timestamp strings to seconds since the Unix epoch (1970-01-01 00:00:00 UTC)
//! with one exact proleptic-Gregorian conversion. Model/NorKyst output and gauge
//! observations therefore land on the same absolute time axis, and the
//! elapsed-time spacing a harmonic fit needs is preserved.

/// Convert a datetime string to seconds since the Unix epoch.
///
/// Accepts a bare number (already seconds), chrono's
/// `YYYY-MM-DD HH:MM:SS[.f] [UTC]`, or RFC3339
/// `YYYY-MM-DDTHH:MM:SS[.f](Z | ±HH:MM | ±HHMM)`. A numeric UTC offset is
/// applied (UTC = local − offset), so `…+00:00` and `…Z` agree. The date→days
/// conversion is exact proleptic Gregorian.
///
/// Out-of-range fields (`2024-02-30`, `25:00`, `:61`) and non-finite numbers
/// (`NaN`, `inf`) are rejected rather than silently rolled over.
pub(crate) fn parse_datetime_seconds(s: &str) -> Result<f64, String> {
    let s = s.trim();
    if let Ok(v) = s.parse::<f64>() {
        return if v.is_finite() {
            Ok(v)
        } else {
            Err(format!("non-finite time {s:?}"))
        };
    }

    // Strip a trailing `Z` / ` UTC` first — otherwise the `T` in "UTC" would be
    // mistaken for the date/time separator.
    let s = match s.strip_suffix('Z') {
        Some(t) => t.trim_end(),
        None => s.strip_suffix("UTC").map(str::trim_end).unwrap_or(s),
    };

    let (date_part, time_part) = if let Some(pos) = s.find('T') {
        (&s[..pos], s[pos + 1..].trim())
    } else if let Some(pos) = s.find(' ') {
        (&s[..pos], s[pos + 1..].trim())
    } else {
        (s, "")
    };

    let d: Vec<&str> = date_part.split('-').collect();
    if d.len() != 3 {
        return Err(format!("unrecognized date {date_part:?}"));
    }
    let year: i64 = d[0].parse().map_err(|_| "bad year".to_string())?;
    let month: i64 = d[1].parse().map_err(|_| "bad month".to_string())?;
    let day: i64 = d[2].parse().map_err(|_| "bad day".to_string())?;
    if !(1..=12).contains(&month) || !(1..=days_in_month(year, month)).contains(&day) {
        return Err(format!("date out of range {date_part:?}"));
    }

    // Peel off the timezone marker, if any, and record the offset to remove.
    let (time_str, offset_secs) = split_timezone(time_part)?;

    let (mut hour, mut minute, mut second) = (0i64, 0i64, 0.0f64);
    if !time_str.is_empty() {
        let t: Vec<&str> = time_str.split(':').collect();
        if t.len() < 2 {
            return Err(format!("unrecognized time {time_str:?}"));
        }
        hour = t[0].parse().map_err(|_| "bad hour".to_string())?;
        minute = t[1].parse().map_err(|_| "bad minute".to_string())?;
        if t.len() > 2 {
            second = t[2].parse().map_err(|_| "bad second".to_string())?;
        }
        // `second` may be 60 for a leap second; NaN fails the range check.
        if t.len() > 3
            || !(0..=23).contains(&hour)
            || !(0..=59).contains(&minute)
            || !(0.0..61.0).contains(&second)
        {
            return Err(format!("time out of range {time_str:?}"));
        }
    }

    let days = days_from_civil(year, month, day);
    let local = days as f64 * 86_400.0 + hour as f64 * 3600.0 + minute as f64 * 60.0 + second;
    Ok(local - offset_secs)
}

/// Split a `HH:MM:SS`-ish time from its trailing timezone marker.
///
/// Returns `(time_without_tz, offset_seconds)`. `Z` / ` UTC` → offset 0;
/// `±HH:MM` / `±HHMM` → the signed offset in seconds.
fn split_timezone(time_part: &str) -> Result<(&str, f64), String> {
    if let Some(t) = time_part.strip_suffix('Z') {
        return Ok((t.trim_end(), 0.0));
    }
    if let Some(t) = time_part.strip_suffix("UTC") {
        return Ok((t.trim_end(), 0.0));
    }
    // A numeric offset's sign sits after the seconds; the time itself has none.
    if let Some(sign_pos) = time_part.rfind(['+', '-']) {
        let (time, off) = time_part.split_at(sign_pos);
        let sign = if off.starts_with('-') { -1.0 } else { 1.0 };
        let off = &off[1..];
        let (oh, om) = match off.split_once(':') {
            Some((h, m)) => (h, m),
            None if off.len() >= 4 => (&off[..2], &off[2..4]),
            None => (off, "0"),
        };
        let oh: u32 = oh.parse().map_err(|_| "bad tz hour".to_string())?;
        let om: u32 = om.parse().map_err(|_| "bad tz minute".to_string())?;
        if oh > 23 || om > 59 {
            return Err(format!("tz offset out of range {off:?}"));
        }
        let offset = f64::from(oh * 3600 + om * 60);
        return Ok((time.trim_end(), sign * offset));
    }
    Ok((time_part, 0.0))
}

/// Number of days in `month` (1–12) of proleptic-Gregorian `year`.
fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Sort `(time, value)` samples by time and drop repeated timestamps.
///
/// The sort is stable and the first sample at each time is kept, so input order
/// decides between duplicates. Duplicates arise when a dataset holds the same
/// instant twice (overlapping grid selections, a month fetched twice); a
/// harmonic fit fed a repeated time sees a zero step and conflicting equations.
pub(crate) fn sort_dedup_by_time<T>(samples: &mut Vec<(f64, T)>) {
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    samples.dedup_by(|b, a| a.0 == b.0);
}

/// Parse CF time `units` (`"<unit> since <reference datetime>"`).
///
/// Returns `(seconds_per_unit, reference_unix_seconds)`, so a stored value `v`
/// is the instant `reference + v·seconds_per_unit` in Unix seconds. Units:
/// seconds, minutes, hours and days (CF/UDUNITS spellings, singular or
/// plural). The reference accepts what [`parse_datetime_seconds`] accepts,
/// including CF's unpadded `1970-1-1 0:0:0`.
///
/// Only the Gregorian calendars are supported; `calendar` is the variable's
/// `calendar` attribute, if any. `noleap`, `360_day` and the like are
/// rejected rather than misread.
#[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
pub(crate) fn parse_cf_time_units(
    units: &str,
    calendar: Option<&str>,
) -> Result<(f64, f64), String> {
    if let Some(cal) = calendar {
        let cal = cal.trim().to_ascii_lowercase();
        if !matches!(
            cal.as_str(),
            "standard" | "gregorian" | "proleptic_gregorian"
        ) {
            return Err(format!("unsupported calendar {cal:?}"));
        }
    }

    let (unit, reference) = units
        .split_once(" since ")
        .ok_or_else(|| format!("time units {units:?} are not \"<unit> since <date>\""))?;
    let seconds_per_unit = match unit.trim().to_ascii_lowercase().as_str() {
        "seconds" | "second" | "secs" | "sec" | "s" => 1.0,
        "minutes" | "minute" | "mins" | "min" => 60.0,
        "hours" | "hour" | "hrs" | "hr" | "h" => 3600.0,
        "days" | "day" | "d" => 86_400.0,
        other => return Err(format!("unsupported time unit {other:?}")),
    };
    let reference = parse_datetime_seconds(reference)
        .map_err(|e| format!("bad reference time in {units:?}: {e}"))?;
    Ok((seconds_per_unit, reference))
}

/// Days from 1970-01-01 to the given proleptic-Gregorian date.
///
/// Howard Hinnant's `days_from_civil` (public-domain), exact for all valid dates.
pub(crate) fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = if m > 2 { m - 3 } else { m + 9 }; // [0, 11]
    let doy = (153 * mp + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Proleptic-Gregorian `(year, month, day)` of `days` since 1970-01-01.
///
/// Howard Hinnant's `civil_from_days`, the inverse of [`days_from_civil`].
pub(crate) fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (
        if m <= 2 {
            yoe + era * 400 + 1
        } else {
            yoe + era * 400
        },
        m,
        d,
    )
}

/// `YYYY-MM-DD HH:MM:SS` (UTC) of a Unix time, to the nearest second.
pub(crate) fn format_utc(unix: f64) -> String {
    let secs = unix.round() as i64;
    let (days, sod) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}",
        sod / 3600,
        sod % 3600 / 60,
        sod % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    #[test]
    fn cf_time_units_scale_and_reference() {
        let (scale, reference) =
            parse_cf_time_units("seconds since 1970-01-01 00:00:00", None).unwrap();
        assert_eq!((scale, reference), (1.0, 0.0));

        // NorKyst v3 / ROMS style
        let (scale, reference) =
            parse_cf_time_units("hours since 2024-01-30 06:00:00", Some("gregorian")).unwrap();
        assert_eq!(scale, 3600.0);
        assert!((reference - 1_706_594_400.0).abs() < TOL);

        let (scale, reference) =
            parse_cf_time_units("days since 1970-1-1 0:0:0", Some("proleptic_gregorian")).unwrap();
        assert_eq!((scale, reference), (86_400.0, 0.0));

        let (_, reference) =
            parse_cf_time_units("seconds since 2024-01-30T06:00:00Z", None).unwrap();
        assert!((reference - 1_706_594_400.0).abs() < TOL);
    }

    #[test]
    fn cf_time_units_rejects_unknown_forms() {
        assert!(parse_cf_time_units("seconds", None).is_err());
        assert!(parse_cf_time_units("fortnights since 1970-01-01", None).is_err());
        assert!(parse_cf_time_units("days since 1970-01-01", Some("noleap")).is_err());
        assert!(parse_cf_time_units("days since 1970-01-01", Some("360_day")).is_err());
        assert!(parse_cf_time_units("days since yesterday", None).is_err());
    }

    #[test]
    fn days_from_civil_reference_points() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1970, 1, 2), 1);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        // 2000-01-01 is 30 years after the epoch, spanning leap years.
        assert_eq!(days_from_civil(2000, 1, 1), 10_957);
        // 2024-01-01
        assert_eq!(days_from_civil(2024, 1, 1), 19_723);
    }

    #[test]
    fn civil_from_days_inverts_days_from_civil() {
        for days in (-800_000..800_000).step_by(997) {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days, "{y}-{m}-{d}");
        }
        assert_eq!(civil_from_days(19_723 + 59), (2024, 2, 29));
    }

    #[test]
    fn format_utc_round_trips_parse() {
        for s in [
            "1970-01-01 00:00:00",
            "2024-02-29 23:59:59",
            "1969-12-31 12:30:00",
        ] {
            assert_eq!(format_utc(parse_datetime_seconds(s).unwrap()), s);
        }
    }

    #[test]
    fn parse_datetime_formats_agree() {
        let secs = 19_723.0 * 86_400.0 + 3600.0; // 2024-01-01 01:00:00 UTC
        let chrono_disp = parse_datetime_seconds("2024-01-01 01:00:00 UTC").unwrap();
        let rfc3339_z = parse_datetime_seconds("2024-01-01T01:00:00Z").unwrap();
        let rfc3339_off = parse_datetime_seconds("2024-01-01T01:00:00+00:00").unwrap();
        let numeric = parse_datetime_seconds(&format!("{secs}")).unwrap();
        assert!((chrono_disp - secs).abs() < TOL);
        assert!((rfc3339_z - secs).abs() < TOL);
        assert!((rfc3339_off - secs).abs() < TOL);
        assert!((numeric - secs).abs() < TOL);
    }

    #[test]
    fn nonzero_offset_is_applied() {
        // 03:00 at +02:00 is 01:00 UTC.
        let utc = 19_723.0 * 86_400.0 + 3600.0;
        let plus2 = parse_datetime_seconds("2024-01-01T03:00:00+02:00").unwrap();
        assert!((plus2 - utc).abs() < TOL);
        // 23:00 at -02:00 is 01:00 UTC (next day).
        let minus2 = parse_datetime_seconds("2023-12-31T23:00:00-02:00").unwrap();
        assert!((minus2 - utc).abs() < TOL);
        // Compact ±HHMM form.
        let compact = parse_datetime_seconds("2024-01-01T03:00:00+0200").unwrap();
        assert!((compact - utc).abs() < TOL);
    }

    #[test]
    fn out_of_range_fields_are_rejected() {
        // Regression: these used to roll over into the next day/month.
        for bad in [
            "2024-02-30T00:00:00Z",
            "2023-02-29T00:00:00Z",
            "2024-04-31T00:00:00Z",
            "2024-01-01T24:00:00Z",
            "2024-01-01T25:61:00Z",
            "2024-01-01T00:60:00Z",
            "2024-01-01T00:00:61Z",
            "2024-01-01T00:00:NaNZ",
            "2024-01-01T00:00:00+24:00",
            "2024-01-01T00:00:00:00Z",
        ] {
            assert!(parse_datetime_seconds(bad).is_err(), "accepted {bad:?}");
        }
        // Leap days that exist are fine.
        assert!(parse_datetime_seconds("2024-02-29T00:00:00Z").is_ok());
        assert!(parse_datetime_seconds("2000-02-29T00:00:00Z").is_ok());
        assert!(parse_datetime_seconds("1900-02-29T00:00:00Z").is_err());
    }

    #[test]
    fn non_finite_numeric_times_are_rejected() {
        for bad in ["NaN", "nan", "inf", "-inf", "infinity"] {
            assert!(parse_datetime_seconds(bad).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn sort_dedup_keeps_first_of_each_time() {
        let mut s = vec![(3.0, 'c'), (1.0, 'a'), (3.0, 'x'), (2.0, 'b'), (1.0, 'y')];
        sort_dedup_by_time(&mut s);
        assert_eq!(s, vec![(1.0, 'a'), (2.0, 'b'), (3.0, 'c')]);
    }
}
