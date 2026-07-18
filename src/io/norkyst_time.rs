//! Shared date/time parsing for the NorKyst ingest readers.
//!
//! Both the text reader (`norkyst_reader`) and the parquet reader
//! (`norkyst_parquet`) convert timestamp strings to seconds since the Unix
//! epoch (1970-01-01 00:00:00 UTC) with an exact proleptic-Gregorian
//! conversion, so the elapsed-time spacing a harmonic fit needs is preserved.

/// Convert a datetime string to seconds since the Unix epoch.
///
/// Accepts a bare number (already seconds), chrono's
/// `YYYY-MM-DD HH:MM:SS[.f] [UTC]`, or RFC3339
/// `YYYY-MM-DDTHH:MM:SS[.f](Z | ±HH:MM | ±HHMM)`. A numeric UTC offset is
/// applied (UTC = local − offset), so `…+00:00` and `…Z` agree. The date→days
/// conversion is exact proleptic Gregorian.
pub(crate) fn parse_datetime_seconds(s: &str) -> Result<f64, String> {
    let s = s.trim();
    if let Ok(v) = s.parse::<f64>() {
        return Ok(v);
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
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
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
        let oh: f64 = oh.parse().map_err(|_| "bad tz hour".to_string())?;
        let om: f64 = om.parse().map_err(|_| "bad tz minute".to_string())?;
        return Ok((time.trim_end(), sign * (oh * 3600.0 + om * 60.0)));
    }
    Ok((time_part, 0.0))
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

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

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
}
