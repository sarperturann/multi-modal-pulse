import numpy as np
from sklearn import svm

__all__ = ['train_boundary', 'project_boundary', 'linear_interpolate']


def train_boundary(latent_codes,
                   scores,
                   chosen_num_or_ratio=0.02,
                   split_ratio=0.7,
                   invalid_value=FileNotFoundError):

    if (not isinstance(latent_codes, np.ndarray) or
            not len(latent_codes.shape) == 2):
        raise ValueError(f'Input `latent_codes` should be with type'
                         f'`numpy.ndarray`, and shape [num_samples, '
                         f'latent_space_dim]!')
    num_samples = latent_codes.shape[0]
    latent_space_dim = latent_codes.shape[1]
    if (not isinstance(scores, np.ndarray) or not len(scores.shape) == 2 or
            not scores.shape[0] == num_samples or not scores.shape[1] == 1):
        raise ValueError(f'Input `scores` should be with type `numpy.ndarray`, and '
                         f'shape [num_samples, 1], where `num_samples` should be '
                         f'exactly same as that of input `latent_codes`!')
    if chosen_num_or_ratio <= 0:
        raise ValueError(f'Input `chosen_num_or_ratio` should be positive, '
                         f'but {chosen_num_or_ratio} received!')

    if invalid_value is not None:
        latent_codes = latent_codes[scores[:, 0] != invalid_value]
        scores = scores[scores[:, 0] != invalid_value]

    sorted_idx = np.argsort(scores, axis=0)[::-1, 0]
    latent_codes = latent_codes[sorted_idx]
    scores = scores[sorted_idx]
    num_samples = latent_codes.shape[0]
    if 0 < chosen_num_or_ratio <= 1:
        chosen_num = int(num_samples * chosen_num_or_ratio)
    else:
        chosen_num = int(chosen_num_or_ratio)
    chosen_num = min(chosen_num, num_samples // 2)

    train_num = int(chosen_num * split_ratio)
    val_num = chosen_num - train_num
    # Positive samples.
    positive_idx = np.arange(chosen_num)
    np.random.shuffle(positive_idx)
    positive_train = latent_codes[:chosen_num][positive_idx[:train_num]]
    positive_val = latent_codes[:chosen_num][positive_idx[train_num:]]
    # Negative samples.
    negative_idx = np.arange(chosen_num)
    np.random.shuffle(negative_idx)
    negative_train = latent_codes[-chosen_num:][negative_idx[:train_num]]
    negative_val = latent_codes[-chosen_num:][negative_idx[train_num:]]
    # Training set.
    train_data = np.concatenate([positive_train, negative_train], axis=0)
    train_label = np.concatenate([np.ones(train_num, dtype=np.int),
                                  np.zeros(train_num, dtype=np.int)], axis=0)

    # Validation set.
    val_data = np.concatenate([positive_val, negative_val], axis=0)
    val_label = np.concatenate([np.ones(val_num, dtype=np.int),
                                np.zeros(val_num, dtype=np.int)], axis=0)

    # Remaining set.
    remaining_num = num_samples - chosen_num * 2
    remaining_data = latent_codes[chosen_num:-chosen_num]
    remaining_scores = scores[chosen_num:-chosen_num]
    decision_value = (scores[0] + scores[-1]) / 2
    remaining_label = np.ones(remaining_num, dtype=np.int)
    remaining_label[remaining_scores.ravel() < decision_value] = 0
    remaining_positive_num = np.sum(remaining_label == 1)
    remaining_negative_num = np.sum(remaining_label == 0)

    clf = svm.SVC(kernel='linear')
    classifier = clf.fit(train_data, train_label)

    if val_num:
        val_prediction = classifier.predict(val_data)
        correct_num = np.sum(val_label == val_prediction)

    if remaining_num:
        remaining_prediction = classifier.predict(remaining_data)
        correct_num = np.sum(remaining_label == remaining_prediction)

    a = classifier.coef_.reshape(1, latent_space_dim).astype(np.float32)
    return a / np.linalg.norm(a)


def project_boundary(primal, *args):

    assert len(primal.shape) == 2 and primal.shape[0] == 1

    if not args:
        return primal
    if len(args) == 1:
        cond = args[0]
        assert (len(cond.shape) == 2 and cond.shape[0] == 1 and
                cond.shape[1] == primal.shape[1])
        new = primal - primal.dot(cond.T) * cond
        return new / np.linalg.norm(new)
    elif len(args) == 2:
        cond_1 = args[0]
        cond_2 = args[1]
        assert (len(cond_1.shape) == 2 and cond_1.shape[0] == 1 and
                cond_1.shape[1] == primal.shape[1])
        assert (len(cond_2.shape) == 2 and cond_2.shape[0] == 1 and
                cond_2.shape[1] == primal.shape[1])
        primal_cond_1 = primal.dot(cond_1.T)
        primal_cond_2 = primal.dot(cond_2.T)
        cond_1_cond_2 = cond_1.dot(cond_2.T)
        alpha = (primal_cond_1 - primal_cond_2 * cond_1_cond_2) / (
            1 - cond_1_cond_2 ** 2 + 1e-8)
        beta = (primal_cond_2 - primal_cond_1 * cond_1_cond_2) / (
            1 - cond_1_cond_2 ** 2 + 1e-8)
        new = primal - alpha * cond_1 - beta * cond_2
        return new / np.linalg.norm(new)
    else:
        for cond_boundary in args:
            assert (len(cond_boundary.shape) == 2 and cond_boundary.shape[0] == 1 and
                    cond_boundary.shape[1] == primal.shape[1])
        cond_boundaries = np.squeeze(np.asarray(args))
        A = np.matmul(cond_boundaries, cond_boundaries.T)
        B = np.matmul(cond_boundaries, primal.T)
        x = np.linalg.solve(A, B)
        new = primal - (np.matmul(x.T, cond_boundaries))
        return new / np.linalg.norm(new)


def linear_interpolate(latent_code,
                       boundary,
                       start_distance=-3.0,
                       end_distance=3.0,
                       steps=10):

    assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
            len(boundary.shape) == 2 and
            boundary.shape[1] == latent_code.shape[-1])

    linspace = np.linspace(start_distance, end_distance, steps)
    if len(latent_code.shape) == 2:
        linspace = linspace - latent_code.dot(boundary.T)
        linspace = linspace.reshape(-1, 1).astype(np.float32)
        return latent_code + linspace * boundary
    if len(latent_code.shape) == 3:
        linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
        return latent_code + linspace * boundary.reshape(1, 1, -1)
    raise ValueError(f'Input `latent_code` should be with shape '
                     f'[1, latent_space_dim] or [1, N, latent_space_dim] for '
                     f'W+ space in Style GAN!\n'
                     f'But {latent_code.shape} is received.')
