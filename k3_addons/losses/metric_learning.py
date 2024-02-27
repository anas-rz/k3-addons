from keras import ops
from k3_addons.api_export import k3_export


@k3_export("k3_addons.losses.pairwise_distance")
def pairwise_distance(feature, squared=False):
    pairwise_distances_squared = ops.add(
        ops.sum(ops.square(feature), axis=1, keepdims=True),
        ops.sum(ops.square(ops.transpose(feature)), axis=0, keepdims=True),
    ) - 2.0 * ops.matmul(feature, ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = ops.sqrt(
            pairwise_distances_squared + ops.cast(error_mask, dtype="float32") * 1e-16
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = ops.multiply(
        pairwise_distances,
        ops.cast(ops.logical_not(error_mask), dtype="float32"),
    )

    num_data = ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = ops.ones_like(pairwise_distances) - ops.diag(
        ops.ones([num_data])
    )
    pairwise_distances = ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def angular_distance(feature):
    # normalize input
    feature = ops.normalize(feature, axis=1)

    # create adjaceny matrix of cosine similarity
    angular_distances = 1 - ops.matmul(feature, ops.transpose(feature))

    # ensure all distances > 1e-16
    angular_distances = ops.maximum(angular_distances, 0.0)

    return angular_distances
