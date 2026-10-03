//! SCRATCH (not for commit): a stratified x–z channel at rest with a cliff
//! inside one element, for the shoreline instability (TODO P1.3).
use std::collections::HashMap;
use std::sync::Arc;

use dg_rs::boundary::Reflective2D;
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::{Bathymetry2D, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::physics::{
    ConstantMixing, EquationOfState, Forcing, Hydrostatic3D, LinearEOS, PhysicsBuilder,
};
use dg_rs::solver::state::Solution3D;
use dg_rs::solver::{StandardLimiter2D, WetDryConfig};
use dg_rs::source::CoriolisSource2D;
use dg_rs::time::ModeSplitIntegrator;
use dg_rs::types::ElementIndex;
use dg_rs::vertical::{SigmaGrid, SongHaidvogelStretching};

const G: f64 = 9.81;
const RHO0: f64 = 1025.0;

fn main() {
    let args: HashMap<String, String> = std::env::args()
        .skip(1)
        .filter_map(|a| {
            a.split_once('=')
                .map(|(k, v)| (k.to_string(), v.to_string()))
        })
        .collect();
    let get = |k: &str, d: f64| args.get(k).map_or(d, |v| v.parse().unwrap());
    let nx = get("nx", 8.0) as usize;
    let dx = get("dx", 1000.0);
    let deep = get("deep", 300.0);
    let shallow = get("shallow", 15.0);
    let cliff = get("cliff", 3.0) as usize; // element holding the cliff
    let steps = get("steps", 400.0) as usize;
    let dt = get("dt", 36.0);
    let nl = get("levels", 20.0) as usize;
    let order = get("order", 2.0) as usize;
    let f = get("f", 1.2e-4);
    let linear = args.contains_key("linear");
    let shape = args.get("shape").map_or("tanh", String::as_str).to_string();

    let length = nx as f64 * dx;
    let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, length, 0.0, dx, nx, 1));
    let ops = Arc::new(DGOperators2D::new(order));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let x0 = cliff as f64 * dx;
    let bed_fn = move |x: f64, _y: f64| -> f64 {
        let t = ((x - x0) / dx).clamp(0.0, 1.0);
        let w = match shape.as_str() {
            "linear" => t,
            _ => 0.5 * (1.0 + ((t - 0.5) * 8.0).tanh()),
        };
        -(deep + (shallow - deep) * w)
    };
    let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, bed_fn));
    let swe = PhysicsBuilder::swe_2d(
        mesh.clone(),
        ops.clone(),
        geom.clone(),
        ShallowWater2D::new(G),
        Reflective2D::new(),
    )
    .with_bathymetry(bathymetry.clone())
    .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
    .with_wet_dry(WetDryConfig::default())
    .with_source(CoriolisSource2D::f_plane(f))
    .build();
    let sigma = Arc::new(SigmaGrid::new(
        nl,
        SongHaidvogelStretching::new(5.0, 0.4, 10.0),
    ));
    let eos = LinearEOS::default();
    let profile = move |z: f64| -> (f64, f64) {
        if linear {
            (eos.t0 + 0.02 * z, eos.s0)
        } else {
            let step = 0.5 * (1.0 + ((z + 15.0) / 4.0).tanh());
            (
                eos.t0 + 4.0 * step + 0.002 * z.min(0.0),
                eos.s0 - 1.5 * step,
            )
        }
    };
    let physics = Hydrostatic3D::new(
        mesh.clone(),
        ops.clone(),
        geom.clone(),
        sigma.clone(),
        bathymetry.clone(),
        Arc::new(CoriolisSource2D::f_plane(f)),
        eos,
        ConstantMixing::new(1e-3, 0.0),
        swe,
        Forcing {
            surface_stress: [0.0, 0.0],
            bottom_stress: [0.0, 0.0],
            surface_buoyancy_flux: 0.0,
        },
        G,
        RHO0,
    );
    let (ne, nn) = (mesh.n_elements, ops.n_nodes);
    let mut state = Solution3D::new(ne, nn, nl);
    for idx in 0..ne * nn {
        let bed = bathymetry.data[idx];
        let eta = bed.max(0.0);
        state.eta.data[idx] = eta;
        for (l, &s) in sigma.sigma_rho().iter().enumerate() {
            let z = eta + s * (eta - bed);
            (state.temp[idx * nl + l], state.salt[idx * nl + l]) = profile(z);
        }
    }
    physics.update_density(&mut state);
    let physics = physics.with_reference_profile(&state, |z| {
        let (t, s) = profile(z);
        eos.compute_density(t, s, z)
    });
    let mut integrator = ModeSplitIntegrator::new();
    let mut t = 0.0;
    let every = (steps / 30).max(1);
    for n in 0..steps {
        physics.update_density(&mut state);
        integrator.step(&mut state, &physics, dt, t);
        physics.post_process(&mut state);
        t += dt;
        if n % every != 0 && n + 1 != steps {
            continue;
        }
        let (mut best, mut at) = (0.0_f64, 0);
        for (i, (u, v)) in state.u.iter().zip(&state.v).enumerate() {
            let s = u.hypot(*v);
            if s.is_nan() || s > best {
                best = s;
                at = i;
            }
        }
        let column = at / nl;
        let (k, node) = (column / nn, column % nn);
        let [x, _] =
            mesh.reference_to_physical(ElementIndex::new(k), ops.nodes_r[node], ops.nodes_s[node]);
        println!(
            "step {n} t {:.2} h: largest {best:.2e} m/s at element {k} node {node} level {} x {:.0} m depth {:.1}",
            t / 3600.0,
            at % nl,
            x,
            state.eta.data[column] - bathymetry.data[column]
        );
        if !best.is_finite() {
            break;
        }
    }
}
