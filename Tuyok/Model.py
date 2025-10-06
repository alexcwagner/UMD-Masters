import json
import numpy as np
from scipy.integrate import quad

class Model(dict):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recalculate()
    
    def recalculate(self): 
        model_in = self.get("model", {})
        layers_in = model_in.get("layers", [])
        if not layers_in:
            raise ValueError("model.layers is required and cannot be empty")
    
        # Extract as arrays for checks
        rho_abs = [float(layer["density"]) for layer in layers_in]
        axes = [tuple(map(float, layer["abc"])) for layer in layers_in]
    
        # Monotonicity checks (densities and each axis)
        
        _check_monotone(rho_abs, "Layer densities", decreasing=True)
        _check_monotone([a for a,_,_ in axes], "a semi-axis")
        _check_monotone([b for _,b,_ in axes], "b semi-axis")
        _check_monotone([c for _,_,c in axes], "c semi-axis")
    
        computed_layers = []
        total_mass = 0.0
        total_Ix = 0.0
        total_Iy = 0.0
        total_Iz = 0.0
    
        rho_prev = 0.0
        increments = []   # store (a, b, c, Δρ)
        
        for i in reversed(range(len(layers_in))):
            
            a, b, c = axes[i]
            rho_i = rho_abs[i]
            delta_rho = rho_i - rho_prev
            rho_prev = rho_i
    
            V = (4.0/3.0) * np.pi * a * b * c
            dM = delta_rho * V
            dIx = (1.0/5.0) * dM * (b*b + c*c)
            dIy = (1.0/5.0) * dM * (a*a + c*c)
            dIz = (1.0/5.0) * dM * (a*a + b*b)
               
            computed_layers.append({
                "density": rho_i,
                "abc": [a, b, c],
                "r_avg": (a*b*c)**(1/3),
                "differential_density": delta_rho,
                "differential_mass": dM,
                "differential_moment_of_inertia": {
                    "Ix": dIx,
                    "Iy": dIy,
                    "Iz": dIz
                },
                "eff_potential": None
            })

            total_mass += dM
            total_Ix += dIx
            total_Iy += dIy
            total_Iz += dIz  
            
            increments.append((a, b, c, delta_rho))

        increments = list(reversed(increments))
        computed_layers = list(reversed(computed_layers))
    
        # Resolve angular momentum / velocity
        L_in = model_in.get("angular_momentum")
        w_in = model_in.get("angular_velocity")
        if w_in is not None:
            omega = float(w_in)
            L = total_Iz * omega
        elif L_in is not None:
            L = float(L_in)
            omega = L / total_Iz if total_Iz > 0 else 0.0
        else:
            raise ValueError("Provide either model.angular_velocity or model.angular_momentum")
    
        T = 0.5 * total_Iz * omega**2
        
        # Gravitational binding energy (ellipsoidal)
        W_self = sum(self_energy(a,b,c,dr) for (a,b,c,dr) in increments)
        W_mut = 0.0
        for i in range(len(increments)):
            ai,bi,ci,dri = increments[i]
            for j in range(i+1, len(increments)):
                aj,bj,cj,drj = increments[j]
                W_mut += mutual_energy(ai,bi,ci,dri, aj,bj,cj,drj)
        W_total = W_self + W_mut
    
        # Precompute grav integrals for equipotential evaluation
        self._grav_cache = []
        for (a,b,c,drho) in increments:
            I0,Ix,Iy,Iz = I0_Ix_Iy_Iz(a,b,c)
            self._grav_cache.append({
                "a":a,"b":b,"c":c,"drho":drho,
                "I0":I0,"Ix":Ix,"Iy":Iy,"Iz":Iz
            })
        eq_errs = []
        eff_list = []
        for k, layer in enumerate(layers_in):
            a,b,c = map(float, layer["abc"])
            pts = [(a,0,0),(0,b,0),(0,0,c)]
            vals = []
            for (x,y,z) in pts:
                vals.append(self._phi_eff_point(x,y,z,k))
            mean_val = np.mean(vals)
            if mean_val != 0.:
                eq_err = np.sqrt(np.mean((vals - mean_val)**2)) / abs(mean_val)
                eq_errs.append(eq_err)
                rms_rel = eq_err
            else:
                rms_rel = None
            eff_list.append({
                "x": float(vals[0]),
                "y": float(vals[1]),
                "z": float(vals[2]),
                "mean": mean_val,
                "rms_rel": rms_rel
            })
        rel_eq_err = float(np.mean(eq_errs)) if eq_errs else None

        for k in range(len(computed_layers)):
            computed_layers[k]["eff_potential"] = eff_list[k]

        model_out = {
            "layers": computed_layers,
            "angular_momentum": L,
            "angular_velocity": omega,
            "mass": total_mass,
            "moment_of_inertia": {
                "Ix": total_Ix,
                "Iy": total_Iy,
                "Iz": total_Iz
            },

            "rotational_kinetic_energy": T,
            "gravitational_binding_energy": W_total,
            "total_energy": T + W_total,
            "rel_equipotential_error": rel_eq_err
        }
        if "units_to_SI" in model_in:
            model_out["units_to_SI"] = model_in["units_to_SI"]
    
        self["model"] = model_out
            

    def _phi_eff_point(self, x, y, z, k):
        """
        Effective potential at (x,y,z) on layer k boundary.
        Includes contributions from all increments up to layer k.
        """
        omega = self["model"]["angular_velocity"]
        phi_g = 0.0
        # Sum increments up to this layer
        for i in range(k+1):
            ci = self._grav_cache[i]
            a,b,c = ci["a"], ci["b"], ci["c"]
            drho = ci["drho"]
            if drho == 0.0:
                continue
            phi_g += (-np.pi * drho * a*b*c *
                      (ci["I0"] - ci["Ix"]*x*x - ci["Iy"]*y*y - ci["Iz"]*z*z))
        phi_cf = 0.5 * omega**2 * (x*x + y*y)
        return phi_g + phi_cf
    
    def layer_axis_potentials(self, k):
        """
        Return Φ_eff at the 3 axis points of layer k.
        """
        a,b,c = map(float, self["model"]["layers"][k]["abc"])
        pts = [(a,0,0), (0,b,0), (0,0,c)]
        return [self._phi_eff_point(x,y,z,k) for (x,y,z) in pts]

    def equipotential_error_axes(self):
        """
        Equipotential error using only 3 axis points per layer.
        RMS deviation normalized by mean per layer, then averaged across layers.
        """
        errs = []
        for k, layer in enumerate(self["model"]["layers"]):
            vals = self.layer_axis_potentials(k)
            mean = np.mean(vals)
            if mean == 0.0:
                continue
            rel_err = np.sqrt(np.mean((vals - mean)**2)) / abs(mean)
            errs.append(rel_err)
        if not errs:
            return 0.0
        return float(np.mean(errs))

def _check_monotone(seq, name, decreasing=False):
    if decreasing:
        for i in range(1, len(seq)):
            if seq[i] > seq[i-1]:
                raise ValueError(f"{name} must be nonincreasing: index "
                                 f"{i-1}->{i} violates monotonicity.")
    else:
        for i in range(1, len(seq)):
            if seq[i] < seq[i-1]:
                raise ValueError(f"{name} must be nondecreasing: index "
                                 f"{i-1}->{i} violates monotonicity.")
    return True

def I0_Ix_Iy_Iz(a, b, c):
    aa, bb, cc = a*a, b*b, c*c
    def Delta(u): return np.sqrt((aa+u)*(bb+u)*(cc+u))
    I0, _ = quad(lambda u: 1.0/Delta(u), 0, np.inf, limit=200)
    Ix, _ = quad(lambda u: 1.0/((aa+u)*Delta(u)), 0, np.inf, limit=200)
    Iy, _ = quad(lambda u: 1.0/((bb+u)*Delta(u)), 0, np.inf, limit=200)
    Iz, _ = quad(lambda u: 1.0/((cc+u)*Delta(u)), 0, np.inf, limit=200)
    return I0, Ix, Iy, Iz

def self_energy(a, b, c, delta_rho):
    V = (4.0/3.0) * np.pi * a*b*c
    M = delta_rho * V
    if M == 0: return 0.0
    I0, _, _, _ = I0_Ix_Iy_Iz(a, b, c)
    return -(3.0/10.0) * M*M * I0

def mutual_energy(ai, bi, ci, dri, aj, bj, cj, drj):
    if dri == 0 or drj == 0:
        return 0.0
    Vi = (4.0/3.0) * np.pi * ai*bi*ci
    Ix_i = Vi * ai*ai / 5.0
    Iy_i = Vi * bi*bi / 5.0
    Iz_i = Vi * ci*ci / 5.0
    I0_j, Ix_j, Iy_j, Iz_j = I0_Ix_Iy_Iz(aj, bj, cj)
    coeff = -np.pi * dri * drj * (aj * bj * cj)
    return coeff * (I0_j*Vi - Ix_j*Ix_i - Iy_j*Iy_i - Iz_j*Iz_i)



if __name__ == '__main__':
    model_json_str = '''
    {
        "model": 
        {
            "layers": 
            [
              { "density": 3.0, "abc": [1.0, 1.0, 1.0] },
              { "density": 2.0, "abc": [2.0, 2.0, 2.0] },
              { "density": 1.0, "abc": [3.0, 3.0, 3.0] }
            ],
            "angular_velocity": 0.3
        },
        "algorithm": 
        {
            "type": "simulated annealing",
            "parameters": 
            {
                "starting_temperature": 1.0,
                "ending_temperature": 1.0e-6,
                "cooling_schedule": 0.99,
                "current_iteration": 0,
                "current_temperature": 1.0,
                "perturbation_scheme": "random_abc_scale ±5%"
            }
        }
    }'''

    maclaurin_json_str = '''    
    {
        "model": 
        {
            "layers": 
            [
                { "density": 1.0, "abc": [1.0, 1.0, 0.8] }
            ],
            "angular_velocity": 0.43
        }
    }'''
    
    jacobi_json_str = '''
    {
        "model": 
        {
            "layers": 
            [
                { "density": 1.0, "abc": [1.0, 0.7, 0.5] }
            ],
            "angular_velocity": 0.62
        }
    }'''
        
    model = Model(json.loads(model_json_str))
    print(json.dumps(model, indent=4))
    
    
    
    
