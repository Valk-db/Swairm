import numpy as np
from sklearn.utils.extmath import randomized_svd
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_non_iid_clients(n_clients, m, n, rank, heterogeneity, rng):
    A_shared = rng.standard_normal((rank, n)) / np.sqrt(n)
    B_shared = rng.standard_normal((m, rank)) / np.sqrt(rank)
    A_list, B_list = [], []
    for _ in range(n_clients):
        A_k = A_shared + heterogeneity * rng.standard_normal((rank, n)) / np.sqrt(n)
        B_k = B_shared + heterogeneity * rng.standard_normal((m, rank)) / np.sqrt(rank)
        A_list.append(A_k)
        B_list.append(B_k)
    return A_list, B_list


def make_dora_clients(n_clients, m_dim, n, rank, heterogeneity, rng):
    A_list, B_list = make_non_iid_clients(n_clients, m_dim, n, rank, heterogeneity, rng)
    m_list = []
    for _ in range(n_clients):
        m_shared = np.ones(m_dim)
        m_k = m_shared + heterogeneity * rng.normal(0, 0.3, m_dim)
        m_k = np.clip(m_k, 0.1, 3.0)
        m_list.append(m_k)
    return A_list, B_list, m_list


def apply_dora_magnitude(B, A, m):
    dense = B @ A
    return dense * m[:, np.newaxis]


def aggregate_magnitudes(m_list, weights=None):
    if weights is None:
        weights = np.ones(len(m_list))
    weights = np.array(weights) / np.sum(weights)
    return np.average(m_list, axis=0, weights=weights)


def dense_reconstructions(A_list, B_list):
    return [B_k @ A_k for A_k, B_k in zip(A_list, B_list)]


def naive_factor_average(A_list, B_list):
    A_avg = np.mean(A_list, axis=0)
    B_avg = np.mean(B_list, axis=0)
    return B_avg @ A_avg


def reconstruct_aggregate_svd(A_list, B_list, target_rank, n_components=None):
    if n_components is None:
        n_components = target_rank
    dense_updates = dense_reconstructions(A_list, B_list)
    dense_avg = np.mean(dense_updates, axis=0)
    U, S, Vt = randomized_svd(dense_avg, n_components=n_components, random_state=42)
    A_global = np.diag(np.sqrt(S[:target_rank])) @ Vt[:target_rank]
    B_global = U[:, :target_rank] @ np.diag(np.sqrt(S[:target_rank]))
    dense_truncated = B_global @ A_global
    return dense_truncated, S, dense_avg


def rel_frobenius_error(estimate, ground_truth):
    return np.linalg.norm(estimate - ground_truth, "fro") / np.linalg.norm(ground_truth, "fro")


def test1_reconstruction_bias():
    print("\n=== Test 1: Reconstruct+SVD vs naive factor averaging ===")
    m, n, rank = 128, 256, 4
    n_clients = 12
    heterogeneity_levels = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    naive_errors, svd_errors = [], []
    for het in heterogeneity_levels:
        A_list, B_list = make_non_iid_clients(n_clients, m, n, rank, het, RNG)
        dense_updates = dense_reconstructions(A_list, B_list)
        ground_truth = np.mean(dense_updates, axis=0)
        naive = naive_factor_average(A_list, B_list)
        svd_trunc, _, _ = reconstruct_aggregate_svd(A_list, B_list, target_rank=rank)
        e_naive = rel_frobenius_error(naive, ground_truth)
        e_svd = rel_frobenius_error(svd_trunc, ground_truth)
        naive_errors.append(e_naive)
        svd_errors.append(e_svd)
        print(f" heterogeneity={het:>4}: naive_factor_avg_err={e_naive:.4f} "
              f"reconstruct_svd_err={e_svd:.4f}")
    return heterogeneity_levels, naive_errors, svd_errors


def test_dora_full_pipeline():
    print("\n=== DoRA Full Pipeline Audit (Post-SVD) ===")
    m_dim, n, rank = 128, 256, 4
    n_clients = 12
    heterogeneity = 1.0
    target_rank = rank

    A_list, B_list, m_list = make_dora_clients(n_clients, m_dim, n, rank, heterogeneity, RNG)

    dense_dora = [apply_dora_magnitude(B, A, m) for B, A, m in zip(B_list, A_list, m_list)]
    ground_truth = np.mean(dense_dora, axis=0)

    m_frozen = np.ones(m_dim)
    dense_frozen = [apply_dora_magnitude(B, A, m_frozen) for B, A in zip(B_list, A_list)]
    frozen_avg = np.mean(dense_frozen, axis=0)
    e_frozen = rel_frobenius_error(frozen_avg, ground_truth)

    m_global = aggregate_magnitudes(m_list)
    dense_no_m = [B @ A for B, A in zip(B_list, A_list)]
    dense_avg_no_m = np.mean(dense_no_m, axis=0)
    U, S, Vt = randomized_svd(dense_avg_no_m, n_components=target_rank, random_state=42)
    A_global = np.diag(np.sqrt(S)) @ Vt
    B_global = U @ np.diag(np.sqrt(S))
    dense_svd = B_global @ A_global
    dense_svd_with_m = dense_svd * m_global[:, np.newaxis]
    e_post_svd = rel_frobenius_error(dense_svd_with_m, ground_truth)

    print(f" Frozen m=1.0 error: {e_frozen:.4f}")
    print(f" Post-SVD m scaling: {e_post_svd:.4f}")
    print(f" m_global mean: {m_global.mean():.3f}, std: {m_global.std():.3f}")

    if e_post_svd < e_frozen * 0.8:
        print(" ✅ Good: Proper magnitude gives clear benefit!")
    else:
        print(" ⚠️ Weak benefit — magnitude handling may need fixing.")


def test_dora_pre_svd():
    print("\n=== DoRA Pre-SVD Magnitude Test ===")
    m_dim, n, rank = 128, 256, 4
    n_clients = 12
    heterogeneity = 1.0
    target_rank = rank

    A_list, B_list, m_list = make_dora_clients(n_clients, m_dim, n, rank, heterogeneity, RNG)

    dense_with_m = [apply_dora_magnitude(B, A, m) for B, A, m in zip(B_list, A_list, m_list)]
    dense_avg = np.mean(dense_with_m, axis=0)

    U, S, Vt = randomized_svd(dense_avg, n_components=target_rank, random_state=42)
    A_global = np.diag(np.sqrt(S)) @ Vt
    B_global = U @ np.diag(np.sqrt(S))
    dense_result = B_global @ A_global

    ground_truth = np.mean(dense_with_m, axis=0)
    e_pre = rel_frobenius_error(dense_result, ground_truth)

    print(f" Pre-SVD magnitude error: {e_pre:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test1_reconstruction_bias()
    test_dora_full_pipeline()
    test_dora_pre_svd()
    print("\nAll tests completed.")