import jax
import jax.numpy as jnp

def minkowski_norm_sq(a_ij, g_ij):
    """
    a_ij: (0,2)-tensor components to norm
    g_ij: (0,2)-metric components defining the Minkowski norm
    """
    g__ij = jnp.linalg.inv(g_ij) ## TODO: precompute
    return jnp.abs(jnp.einsum("ij, kl, ik, jl -> ", a_ij, a_ij, g__ij, g__ij)) ## TODO: negativity abs or sq?!


def norm_papuc(v__i, g_ij, z__i):
    """
    Papuc's norm for a tangent vector v__i in a Lorentzian manifold.
    
    Args:
        v__i: (n,) vector
        g_ij: (n, n) symmetric Lorentzian metric (signature (-,+,+,...))
        z__i: (n,) reference future-directed timelike vector (g(z,z) < 0)

    Returns:
        Scalar norm ||V||_Z
    """
    # Inner products
    gzz = z__i.T @ g_ij @ z__i  # g(z, z)
    gvv = v__i.T @ g_ij @ v__i  # g(v, v)
    gzv = v__i.T @ g_ij @ z__i  # g(z, v)
    # print(gzz, gvv, gzv)
    # assert gzz < 0, "z is not timelike"
    
    return jnp.abs((jnp.abs(gzv) + jnp.sqrt(gzv**2 - gzz*gvv + 1e-6))/(-jnp.abs(gzz) - 1e-8))


def sample_points_2_sphere(N, key):
    key_theta, key_phi = jax.random.split(key)
    
    # Sample uniformly distributed azimuthal angle θ in [0, 2π)
    theta = jax.random.uniform(key_theta, shape=(N,), minval=0.0, maxval=2 * jnp.pi)
    
    # Sample z uniformly in [-1, 1] and compute polar angle φ
    z = jax.random.uniform(key_phi, shape=(N,), minval=-1.0, maxval=1.0)
    r = jnp.sqrt(1.0 - z**2)

    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)

    return jnp.stack([x, y, z], axis=1)  # shape (N, 3)


def norm_papuc_operator(a__i_j, g_ij, z__i, key, sampling="boundary", N=1_000_000, debug=False):
    """
    a__i_j: rank (1,1) tensor a^i_j with upper index i (first, a__i_j[i,:]) and lower index j (second, a__i_j[:,j])
    g_ij: rank (0,2) metric tensor
    z__i: the future time-oriented reference vector
    """
    
    key, subkey = jax.random.split(key)
    
    if sampling=="cone":
        x__i_batch = jax.random.uniform(subkey, (N, 4), minval=-1, maxval=1)
        x__i_batch = x__i_batch / jax.vmap(norm_papuc, in_axes=[0, None, None])(x__i_batch, g_ij, z__i)[:,None]
    
    elif sampling=="boundary":
        if debug:
            assert jnp.allclose(jnp.diag(g_ij), jnp.array([-1,1,1,1])), "Boundary sampling only works for normal-form metrics"
        # Sample N points uniformly from a 2-sphere
        x__i_batch = jnp.hstack((
            jnp.zeros([N,1]),
            sample_points_2_sphere(N, key)
        ))
        x__i_batch = jnp.vstack([
            z__i, ## apex
            x__i_batch ## base
            ])
    
    def ax_norm(x__i):
        ax__i = jnp.einsum("ji,j->i", a__i_j, x__i)
        return norm_papuc(ax__i, g_ij, z__i)
        
    ax_norms = jax.vmap(ax_norm)(x__i_batch)
    ## take supremum of transformed norms
    idx = jnp.argmax(ax_norms)
    return ax_norms[idx]


# Define the loss function
def divergence_papuc_operator_canonical(a_ij, g_ij, z__i, key, debug=False):
    ## TODO: take care of z__i
    a_ij_org = a_ij
    g_ij_org = g_ij
    ### 1) Bring metric in canonical form
    ## Eigendecomposition
    eigvals, eigvecs = jnp.linalg.eigh(g_ij_org)
    if debug:
        assert jnp.allclose(g_ij_org, eigvecs @ jnp.diag(eigvals) @ eigvecs.T), "The diagonalization G = U \Lambda U^T does not hold"

    ### 2) Project the a_ij tensor into canonical coordinate system
    ## Aliases
    L = jnp.diag(jnp.abs(eigvals))
    L_inv = jnp.linalg.inv(L)
    L_sqrt = jnp.sqrt(L)
    L_inv_sqrt = jnp.sqrt(L_inv)
    U = eigvecs

    ## Canonical
    M = jnp.diag(jnp.array([-1,1,1,1])) # Minkowski
    V = U @ L_sqrt ## same as eigvecs_scaled = eigvecs * jnp.abs(eigvals)**.5 where we scale each column
    g_ij_can = M
    if debug:
        assert jnp.allclose(g_ij_org, V @ g_ij_can @ V.T), "The diagonalization G = V M V^T does not hold"
    
    org_to_can = lambda A : U.T @ A @ U @ L_inv
    if debug:
        can_to_org = lambda B : U @ B @ L @ U.T
        assert jnp.allclose(org_to_can(g_ij_org), M), "org_to_can transformation does not hold for the metric"
        assert jnp.allclose(can_to_org(M), g_ij_org), "can_to_org transformation does not hold for the metric"
        assert jnp.allclose(can_to_org(org_to_can(a_ij_org)), a_ij_org), "forward and inverse transformation is not the identity for an arbitrary tensor"
    a_ij_can = org_to_can(a_ij_org)

    ### 3) Project the z__i into canonical coordinate system
    ## TODO: use a fixed z__i? As long as it's timelike [\alpha,0,0,0], we can rescale by \alpha
    if debug:
        org_to_can_vec = lambda v : L_sqrt @ U.T @ v
        can_to_org_vec = lambda w : U @ L_inv_sqrt @ w
        assert jnp.allclose(can_to_org_vec(org_to_can_vec(v_i_org)), v_i_org), "forward and inverse transformation is not the identity for an arbitrary vector"
    z__i_can = jnp.array([1,0,0,0])

    ### 4) Bring tensor in (1,1) form through inverse metric components (2,0) (identical to (0,2) metric components in canonical form!)
    g__ij_can = g_ij_can ## Minkowski^-1 = Minkowski
    a__i_j_can = jnp.einsum("ij,jk -> ik", a_ij_can, g__ij_can)
    delta__i_j_can = a__i_j_can - jnp.eye(4) ## TODO: are we sure this doesn't find similar matrix?
    
    ## Same:
    # delta_ij_can = a_ij_can - g_ij_can
    # delta__i_j_can = jnp.einsum("ij,jk -> ik", delta_ij_can, g__ij_can)
    ## Since all of the above operations are linear, we should be able to pass 

    
    ### 5) Compute operator norm through base sampling (compare to cone sampling to see if variance is the issue)
    return norm_papuc_operator(delta__i_j_can, g_ij_can, z__i_can, key, N=1_000, sampling="boundary")