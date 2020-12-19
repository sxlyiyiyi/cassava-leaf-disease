import tensorflow as tf


# 随机缩放
@tf.function
def random_zoom(img, label):
    scale = tf.random.uniform([], 0.85, 1.15)
    img_h, img_w, img_channels = tf.shape(img)[0], tf.shape(img)[1], tf.shape(img)[2]
    img_h_new, img_w_new = tf.cast(tf.cast(img_h, tf.float32) * scale, tf.int32), tf.cast(tf.cast(img_w, tf.float32) * scale, tf.int32)
    img_h_diff, img_w_diff = tf.math.abs(img_h_new - img_h), tf.math.abs(img_w_new - img_w)
    img_t, img_l = img_h_diff // 2, img_w_diff // 2
    img_b, img_r = img_h_diff - img_t, img_w_diff - img_l
    img_new = tf.image.resize(img, [img_h_new, img_w_new], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return tf.cond(scale >= 1.0,
                   lambda: img_new[img_t:img_t + img_h, img_l:img_l + img_w, :],
                   lambda: tf.pad(img_new, [[img_t, img_b], [img_l, img_r], [0, 0]])), label


def random_rotate(img, label):
    pass


def rotate(images,
           angles,
           interpolation="NEAREST",
           name=None):
    """Rotate image(s) counterclockwise by the passed angle(s) in radians.

    Args:
      images: A tensor of shape
        `(num_images, num_rows, num_columns, num_channels)`
        (NHWC), `(num_rows, num_columns, num_channels)` (HWC), or
        `(num_rows, num_columns)` (HW).
      angles: A scalar angle to rotate all images by, or (if `images` has rank 4)
        a vector of length num_images, with an angle for each image in the
        batch.
      interpolation: Interpolation mode. Supported values: "NEAREST",
        "BILINEAR".
      name: The name of the op.

    Returns:
      Image(s) with the same type and shape as `images`, rotated by the given
      angle(s). Empty space due to the rotation will be filled with zeros.

    Raises:
      TypeError: If `images` is an invalid type.
    """
    with tf.name_scope(name or "rotate"):
        image_or_images = tf.convert_to_tensor(images)
        if image_or_images.dtype.base_dtype not in _IMAGE_DTYPES:
            raise TypeError("Invalid dtype %s." % image_or_images.dtype)
        images = img_utils.to_4D_image(image_or_images)
        original_ndims = img_utils.get_ndims(image_or_images)

        image_height = tf.cast(tf.shape(images)[1], tf.dtypes.float32)[None]
        image_width = tf.cast(tf.shape(images)[2], tf.dtypes.float32)[None]
        output = transform(
            images,
            angles_to_projective_transforms(angles, image_height, image_width),
            interpolation=interpolation,
        )
        return from_4D_image(output, original_ndims)


def angles_to_projective_transforms(angles,
                                    image_height,
                                    image_width,
                                    name=None):
    """Returns projective transform(s) for the given angle(s).

    Args:
      angles: A scalar angle to rotate all images by, or (for batches of
        images) a vector with an angle to rotate each image in the batch. The
        rank must be statically known (the shape is not `TensorShape(None)`.
      image_height: Height of the image(s) to be transformed.
      image_width: Width of the image(s) to be transformed.
      name:

    Returns:
      A tensor of shape (num_images, 8). Projective transforms which can be
      given to `transform` op.
    """
    with tf.name_scope(name or "angles_to_projective_transforms"):
        angle_or_angles = tf.convert_to_tensor(
            angles, name="angles", dtype=tf.dtypes.float32
        )
        if len(angle_or_angles.get_shape()) == 0:
            angles = angle_or_angles[None]
        elif len(angle_or_angles.get_shape()) == 1:
            angles = angle_or_angles
        else:
            raise ValueError("angles should have rank 0 or 1.")
        cos_angles = tf.math.cos(angles)
        sin_angles = tf.math.sin(angles)
        x_offset = (
            (image_width - 1)
            - (cos_angles * (image_width - 1) - sin_angles * (image_height - 1))
        ) / 2.0
        y_offset = (
            (image_height - 1)
            - (sin_angles * (image_width - 1) + cos_angles * (image_height - 1))
        ) / 2.0
        num_angles = tf.shape(angles)[0]
        return tf.concat(
            values=[
                cos_angles[:, None],
                -sin_angles[:, None],
                x_offset[:, None],
                sin_angles[:, None],
                cos_angles[:, None],
                y_offset[:, None],
                tf.zeros((num_angles, 2), tf.dtypes.float32),
            ],
            axis=1,
        )


def from_4D_image(image, ndims):
    """Convert back to an image with `ndims` rank.

    Args:
      image: 4D `Tensor`.
      ndims: The original rank of the image.

    Returns:
      `ndims`-D `Tensor` with the same type.
    """
    with tf.control_dependencies(
        [tf.debugging.assert_rank(image, 4, message="`image` must be 4D tensor")]
    ):
        if isinstance(ndims, tf.Tensor):
            return _dynamic_from_4D_image(image, ndims)
        elif ndims == 2:
            return tf.squeeze(image, [0, 3])
        elif ndims == 3:
            return tf.squeeze(image, [0])
        else:
            return image


def _dynamic_from_4D_image(image, original_rank):
    shape = tf.shape(image)
    # 4D image <= [N, H, W, C] or [N, C, H, W]
    # 3D image <= [1, H, W, C] or [1, C, H, W]
    # 2D image <= [1, H, W, 1]
    begin = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    end = 4 - tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = shape[begin:end]
    return tf.reshape(image, new_shape)


def transform(images,
              transforms,
              interpolation="NEAREST",
              output_shape=None,
              name=None):
    """Applies the given transform(s) to the image(s).

    Args:
      images: A tensor of shape (num_images, num_rows, num_columns,
        num_channels) (NHWC), (num_rows, num_columns, num_channels) (HWC), or
        (num_rows, num_columns) (HW).
      transforms: Projective transform matrix/matrices. A vector of length 8 or
        tensor of size N x 8. If one row of transforms is
        [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the *output* point
        `(x, y)` to a transformed *input* point
        `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
        where `k = c0 x + c1 y + 1`. The transforms are *inverted* compared to
        the transform mapping input points to output points. Note that
        gradients are not backpropagated into transformation parameters.
      interpolation: Interpolation mode.
        Supported values: "NEAREST", "BILINEAR".
      output_shape: Output dimesion after the transform, [height, width].
        If None, output is the same size as input image.

      name: The name of the op.

    Returns:
      Image(s) with the same type and shape as `images`, with the given
      transform(s) applied. Transformed coordinates outside of the input image
      will be filled with zeros.

    Raises:
      TypeError: If `image` is an invalid type.
      ValueError: If output shape is not 1-D int32 Tensor.
    """
    with tf.name_scope(name or "transform"):
        image_or_images = tf.convert_to_tensor(images, name="images")
        transform_or_transforms = tf.convert_to_tensor(
            transforms, name="transforms", dtype=tf.dtypes.float32
        )
        if image_or_images.dtype.base_dtype not in _IMAGE_DTYPES:
            raise TypeError("Invalid dtype %s." % image_or_images.dtype)
        images = img_utils.to_4D_image(image_or_images)
        original_ndims = img_utils.get_ndims(image_or_images)

        if output_shape is None:
            output_shape = tf.shape(images)[1:3]

        output_shape = tf.convert_to_tensor(
            output_shape, tf.dtypes.int32, name="output_shape"
        )

        if not output_shape.get_shape().is_compatible_with([2]):
            raise ValueError(
                "output_shape must be a 1-D Tensor of 2 elements: "
                "new_height, new_width"
            )

        if len(transform_or_transforms.get_shape()) == 1:
            transforms = transform_or_transforms[None]
        elif transform_or_transforms.get_shape().ndims is None:
            raise ValueError("transforms rank must be statically known")
        elif len(transform_or_transforms.get_shape()) == 2:
            transforms = transform_or_transforms
        else:
            transforms = transform_or_transforms
            raise ValueError(
                "transforms should have rank 1 or 2, but got rank %d"
                % len(transforms.get_shape())
            )

        output = tf.raw_ops.ImageProjectiveTransformV2(
            images=images,
            transforms=transforms,
            output_shape=output_shape,
            interpolation=interpolation.upper(),
        )
        return from_4D_image(output, original_ndims)

