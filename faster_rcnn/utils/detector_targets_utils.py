import tensorflow as tf

from ..bbox_utils import bbox_overlap_tf, encode


def generate_detector_targets(
    proposals, gt_bboxes, bg_low, bg_high, fg_low, pos_prop_perc, prop_batch
):
    overlaps = bbox_overlap_tf(proposals, gt_bboxes[:, :4])
    proposal_label = tf.fill(len(proposals), -1)

    # max overlap with a gt_bbox for each proposal
    max_overlaps = tf.reduce_max(overlaps, axis=1)

    # background proposals have label 0
    bg_filter_low = max_overlaps >= bg_low
    bg_filter_high = max_overlaps < bg_high
    bg_filter = tf.logical_and(bg_filter_low, bg_filter_high)
    proposal_label = tf.where(bg_filter, tf.zeros_like(proposal_label), proposal_label)

    # label of best ground truth box for each proposal
    best_gt_bbox = tf.argmax(overlaps, axis=1)
    labels_for_proposals = tf.gather(gt_bboxes[:, -1], best_gt_bbox) + 1
    labels_for_proposals = tf.cast(labels_for_proposals, tf.int32)
    fg_filter = max_overlaps >= fg_low
    proposal_label = tf.where(fg_filter, labels_for_proposals, proposal_label)

    # best proposal for each ground truth box
    # (would have ground truth box label)
    best_proposal_for_gt_bbox = tf.argmax(overlaps, axis=0)
    label_for_best_proposal = tf.sparse.SparseTensor(
        tf.reshape(best_proposal_for_gt_bbox, [-1, 1]),
        gt_bboxes[:, -1] + 1,
        [len(proposal_label)],
    )
    label_for_best_proposal = tf.sparse.to_dense(
        label_for_best_proposal, validate_indices=False, default_value=0
    )
    best_proposal_for_gtbbox_filter = label_for_best_proposal > 0
    label_for_best_proposal = tf.cast(label_for_best_proposal, tf.int32)
    proposal_label = tf.where(
        best_proposal_for_gtbbox_filter, label_for_best_proposal, proposal_label
    )

    # number of proposal should not be larger than the minibatch size
    # disable some fg
    num_fg_thresh = pos_prop_perc * prop_batch
    fg_ind = tf.where(proposal_label > 0)
    num_fg = len(fg_ind)
    if num_fg > num_fg_thresh:
        # reduce number of fg proposals
        num_indices_to_disable = tf.cast(num_fg - num_fg_thresh, tf.int32)
        fg_ind = tf.random.shuffle(fg_ind)
        fg_ind_to_disable = fg_ind[:num_indices_to_disable]
        fg_ind_to_disable_filter = tf.sparse.SparseTensor(
            tf.reshape(fg_ind_to_disable, [-1, 1]),
            [True] * len(fg_ind_to_disable),
            [len(proposal_label)],
        )

        fg_ind_to_disable_filter = tf.sparse.to_dense(
            fg_ind_to_disable_filter, validate_indices=False, default_value=False
        )
        proposal_label = tf.where(
            fg_ind_to_disable_filter, tf.fill(proposal_label.shape, -1), proposal_label
        )

        num_fg = num_fg_thresh

    # disable some bg
    num_bg = len(proposal_label) - num_fg
    num_bg_thresh = prop_batch - num_fg_thresh
    if num_bg > num_bg_thresh:
        bg_ind = tf.where(proposal_label == 0)
        num_indices_to_disable = tf.cast(num_bg - num_bg_thresh, tf.int32)
        bg_ind = tf.random.shuffle(bg_ind)
        bg_ind_to_disable = bg_ind[:num_indices_to_disable]
        bg_ind_to_disable_filter = tf.sparse.SparseTensor(
            tf.reshape(bg_ind_to_disable, [-1, 1]),
            [True] * len(bg_ind_to_disable),
            [len(proposal_label)],
        )
        bg_ind_to_disable_filter = tf.sparse.to_dense(
            bg_ind_to_disable_filter, validate_indices=False, default_value=False
        )
        proposal_label = tf.where(
            bg_ind_to_disable_filter, tf.fill(proposal_label.shape, -1), proposal_label
        )

    # Calculate proposals bounding box targets using the ground truth boxes
    proposals_with_label_filter = proposal_label > 0
    proposals_with_label = tf.where(proposals_with_label_filter)
    proposals_with_label = tf.reshape(proposals_with_label, [-1])
    gt_bbox_ind_for_proposals = tf.gather(best_gt_bbox, proposals_with_label)

    gt_bbox_for_proposals = tf.gather(gt_bboxes[:, :-1], gt_bbox_ind_for_proposals)
    boxes_for_proposal = tf.gather(proposals, proposals_with_label)
    # encode proposals using ground_truth boxes
    proposal_targets = encode(boxes_for_proposal, gt_bbox_for_proposals)

    # for shape compatibility
    proposal_targets = tf.scatter_nd(
        proposals_with_label[:, None],
        proposal_targets,
        tf.cast(proposals.shape, tf.int64),
    )

    # only return tensor containing only valid boxes and labels
    # (negative labels are ignored)
    filter_ = proposal_label >= 0
    proposal_targets = tf.boolean_mask(proposal_targets, filter_)
    proposal_label = tf.boolean_mask(proposal_label, filter_)
    proposals = tf.boolean_mask(proposals, filter_)

    return proposal_targets, proposal_label, proposals
