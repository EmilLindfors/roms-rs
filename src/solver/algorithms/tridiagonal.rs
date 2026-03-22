//! Tridiagonal matrix solver (Thomas algorithm).
//!
//! Used for implicit vertical diffusion.

/// Solve a tridiagonal system Ax = d using the Thomas algorithm.
///
/// The system is defined by:
/// a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
///
/// # Arguments
///
/// * `a` - Lower diagonal (length n). a[0] is ignored.
/// * `b` - Main diagonal (length n).
/// * `c` - Upper diagonal (length n). c[n-1] is ignored.
/// * `d` - Right-hand side (length n).
/// * `x` - Solution vector (length n).
/// * `c_prime` - Temporary buffer (length n).
/// * `d_prime` - Temporary buffer (length n).
///
/// # Panics
///
/// Panics if array lengths do not match.
///
/// # Stability
///
/// This algorithm is stable if the matrix is diagonally dominant or symmetric positive definite.
/// For implicit diffusion, the matrix is usually diagonally dominant.
pub fn solve_tridiagonal(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
    x: &mut [f64],
    c_prime: &mut [f64],
    d_prime: &mut [f64],
) {
    let n = b.len();
    assert_eq!(a.len(), n, "a length mismatch");
    assert_eq!(c.len(), n, "c length mismatch");
    assert_eq!(d.len(), n, "d length mismatch");
    assert_eq!(x.len(), n, "x length mismatch");
    assert_eq!(c_prime.len(), n, "c_prime length mismatch");
    assert_eq!(d_prime.len(), n, "d_prime length mismatch");

    if n == 0 {
        return;
    }

    // Forward elimination
    let denom = b[0];
    // Avoid division by zero? Usually b[0] != 0 for diffusion.
    if denom.abs() < 1e-15 {
        // Handle singularity or just panic?
        // For diffusion (I - dt*D), diagonal is 1 + something positive. Always > 1.
        // So safe.
    }
    
    c_prime[0] = c[0] / denom;
    d_prime[0] = d[0] / denom;

    for i in 1..n {
        let m = 1.0 / (b[i] - a[i] * c_prime[i - 1]);
        if i < n - 1 {
            c_prime[i] = c[i] * m;
        }
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) * m;
    }

    // Back substitution
    x[n - 1] = d_prime[n - 1];

    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
}

/// Solve a tridiagonal system Ax = d using the Thomas algorithm (allocating version).
pub fn solve_tridiagonal_alloc(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];
    
    solve_tridiagonal(a, b, c, d, &mut x, &mut c_prime, &mut d_prime);
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_tridiagonal_identity() {
        // Identity matrix
        let n = 5;
        let a = vec![0.0; n];
        let b = vec![1.0; n];
        let c = vec![0.0; n];
        let d = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let x = solve_tridiagonal_alloc(&a, &b, &c, &d);
        
        for i in 0..n {
            assert!((x[i] - d[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_solve_tridiagonal_simple() {
        // System:
        // 2x0 - 1x1 = 1
        // -1x0 + 2x1 - 1x2 = 0
        // -1x1 + 2x2 = 1
        // Solution: x = [1, 1, 1]
        
        let n = 3;
        let a = vec![0.0, -1.0, -1.0]; // a[0] ignored
        let b = vec![2.0, 2.0, 2.0];
        let c = vec![-1.0, -1.0, 0.0]; // c[2] ignored
        let d = vec![1.0, 0.0, 1.0];
        
        let x = solve_tridiagonal_alloc(&a, &b, &c, &d);
        
        assert!((x[0] - 1.0).abs() < 1e-12, "x[0] = {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-12, "x[1] = {}", x[1]);
        assert!((x[2] - 1.0).abs() < 1e-12, "x[2] = {}", x[2]);
    }
}
