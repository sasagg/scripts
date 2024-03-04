

def _process_batch(sess, original_images, semantic_predictions,gt, image_names,
                   image_heights, image_widths, image_id_offset, save_dir,
                   raw_save_dir, ground_truth_dir, train_id_to_eval_id=None):
    """Evaluates and saves one single batch qualitatively.

    Args:
        sess: TensorFlow session.
        original_images: One batch of original images.
        semantic_predictions: One batch of semantic segmentation predictions.
        image_names: Image names.
        image_heights: Image heights.
        image_widths: Image widths.
        image_id_offset: Image id offset for indexing images.
        save_dir: The directory where the predictions will be saved.
        raw_save_dir: The directory where the raw predictions will be saved.
        ground_truth_dir: The directory where the ground truth images will be saved.
        train_id_to_eval_id: A list mapping from train id to eval id.
    """
    (original_images,
     semantic_predictions,
     gt,
     image_names,
     image_heights,
     image_widths) = sess.run([original_images, semantic_predictions,gt,
                               image_names, image_heights, image_widths])

    num_image = semantic_predictions.shape[0]
    for i in range(num_image):
        image_height = np.squeeze(image_heights[i])
        image_width = np.squeeze(image_widths[i])
        original_image = np.squeeze(original_images[i])
        semantic_prediction = np.squeeze(semantic_predictions[i])
        gt_image = np.squeeze(gt[i])
        crop_semantic_prediction = semantic_prediction[:image_height, :image_width]
        # Save image.
        # save_annotation.save_annotation(
        #     original_image, save_dir, _IMAGE_FORMAT % (image_id_offset + i),
        #     add_colormap=False)


        #image_name = os.path.basename(image_names[i].decode('utf-8'))
        #gt_save_path = os.path.join(save_dir, image_name + "_gt")
        #gt_save_path = os.path.join(save_dir, image_name + "_gt.png")

        save_annotation.save_annotation(
            original_image, save_dir, os.path.basename(image_names[i]).decode('utf-8'),
            add_colormap=False)

        save_annotation.save_annotation(
            gt_image[:image_height, :image_width], save_dir,
            os.path.basename(image_names[i]).decode('utf-8')+"_gt", add_colormap=True,
            colormap_type=FLAGS.colormap_type)

        save_annotation.save_annotation(
            crop_semantic_prediction, save_dir,
            os.path.basename(image_names[i]).decode('utf-8')+"_pred", add_colormap=True,
            colormap_type=FLAGS.colormap_type)


        # save_annotation.save_annotation(
        #     gt_image[:image_height, :image_width], save_dir,
        #     (os.path.basename(image_names[i]).decode('utf-8')+"_gt").encode('latin-1'), add_colormap=True,
        #     colormap_type=FLAGS.colormap_type)

        # save_annotation.save_annotation(
        #     crop_semantic_prediction, save_dir,
        #     (os.path.basename(image_names[i]).decode('utf-8')+"_pred").encode('latin-1'), add_colormap=True,
        #     colormap_type=FLAGS.colormap_type)
        
        
      
        if FLAGS.also_save_raw_predictions:
            image_filename = os.path.basename(image_names[i])

            if train_id_to_eval_id is not None:
                crop_semantic_prediction = _convert_train_id_to_eval_id(
                    crop_semantic_prediction,
                    train_id_to_eval_id)
            # save_annotation.save_annotation(
            #     crop_semantic_prediction, raw_save_dir, image_filename,
            #     add_colormap=False)
            save_annotation.save_annotation(
                crop_semantic_prediction, raw_save_dir, image_filename,
                add_colormap=True)
            
            
        


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Get dataset-dependent information.
  dataset = data_generator.Dataset(
      dataset_name=FLAGS.dataset,
      split_name=FLAGS.vis_split,
      dataset_dir=FLAGS.dataset_dir,
      batch_size=FLAGS.vis_batch_size,
      crop_size=[int(sz) for sz in FLAGS.vis_crop_size],
      min_resize_value=FLAGS.min_resize_value,
      max_resize_value=FLAGS.max_resize_value,
      resize_factor=FLAGS.resize_factor,
      model_variant=FLAGS.model_variant,
      is_training=False,
      should_shuffle=False,
      should_repeat=False)

  train_id_to_eval_id = None
  if dataset.dataset_name == data_generator.get_cityscapes_dataset_name():
    tf.logging.info('Cityscapes requires converting train_id to eval_id.')
    train_id_to_eval_id = _CITYSCAPES_TRAIN_ID_TO_EVAL_ID

  # Prepare for visualization.
  tf.gfile.MakeDirs(FLAGS.vis_logdir)
  save_dir = os.path.join(FLAGS.vis_logdir, _SEMANTIC_PREDICTION_SAVE_FOLDER)
  tf.gfile.MakeDirs(save_dir)
  raw_save_dir = os.path.join(
      FLAGS.vis_logdir, _RAW_SEMANTIC_PREDICTION_SAVE_FOLDER)
  tf.gfile.MakeDirs(raw_save_dir)

  ground_truth_dir = os.path.join(FLAGS.vis_logdir, "ground_truth")
  tf.gfile.MakeDirs(ground_truth_dir)

  tf.logging.info('Visualizing on %s set', FLAGS.vis_split)

  with tf.Graph().as_default():
    samples = dataset.get_one_shot_iterator().get_next()

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_of_classes},
        crop_size=[int(sz) for sz in FLAGS.vis_crop_size],
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    if tuple(FLAGS.eval_scales) == (1.0,):
      tf.logging.info('Performing single-scale test.')
      predictions = model.predict_labels(
          samples[common.IMAGE],
          model_options=model_options,
          image_pyramid=FLAGS.image_pyramid)
    else:
      tf.logging.info('Performing multi-scale test.')
      if FLAGS.quantize_delay_step >= 0:
        raise ValueError(
            'Quantize mode is not supported with multi-scale test.')
      predictions = model.predict_labels_multi_scale(
          samples[common.IMAGE],
          model_options=model_options,
          eval_scales=FLAGS.eval_scales,
          add_flipped_images=FLAGS.add_flipped_images)
    predictions = predictions[common.OUTPUT_TYPE]

    if FLAGS.min_resize_value and FLAGS.max_resize_value:
      # Only support batch_size = 1, since we assume the dimensions of original
      # image after tf.squeeze is [height, width, 3].
      assert FLAGS.vis_batch_size == 1

      # Reverse the resizing and padding operations performed in preprocessing.
      # First, we slice the valid regions (i.e., remove padded region) and then
      # we resize the predictions back.
      original_image = tf.squeeze(samples[common.ORIGINAL_IMAGE])
      original_image_shape = tf.shape(original_image)
      predictions = tf.slice(
          predictions,
          [0, 0, 0],
          [1, original_image_shape[0], original_image_shape[1]])
      resized_shape = tf.to_int32([tf.squeeze(samples[common.HEIGHT]),
                                   tf.squeeze(samples[common.WIDTH])])
      predictions = tf.squeeze(
          tf.image.resize_images(tf.expand_dims(predictions, 3),
                                 resized_shape,
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                 align_corners=True), 3)

    tf.train.get_or_create_global_step()
    if FLAGS.quantize_delay_step >= 0:
      contrib_quantize.create_eval_graph()

    num_iteration = 0
    max_num_iteration = FLAGS.max_number_of_iterations

    # checkpoints_iterator = contrib_training.checkpoints_iterator(
    #     FLAGS.checkpoint_dir, min_interval_secs=FLAGS.eval_interval_secs)
    #for checkpoint_path in checkpoints_iterator:
    checkpoint_path = FLAGS.checkpoint_dir  # 事前学習済みモデルのパスを指定
    tf.logging.info(
        'Starting visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                    time.gmtime()))
    tf.logging.info('Visualizing with model %s', checkpoint_path)

    scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold,
        master=FLAGS.master,
        checkpoint_filename_with_path=checkpoint_path)
    with tf.train.MonitoredSession(
        session_creator=session_creator, hooks=None) as sess:
      batch = 0
      image_id_offset = 0
      #print(samples)
      while not sess.should_stop():
        tf.logging.info('Visualizing batch %d', batch + 1)
        _process_batch(sess=sess,
                        original_images=samples[common.ORIGINAL_IMAGE],
                        semantic_predictions=predictions,
                        gt = samples[common.LABEL],
                        image_names=samples[common.IMAGE_NAME],
                        image_heights=samples[common.HEIGHT],
                        image_widths=samples[common.WIDTH],
                        image_id_offset=image_id_offset,
                        save_dir=save_dir,
                        raw_save_dir=raw_save_dir,
                        ground_truth_dir=ground_truth_dir,
                        train_id_to_eval_id=train_id_to_eval_id)
        image_id_offset += FLAGS.vis_batch_size
        batch += 1

    tf.logging.info(
        'Finished visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                      time.gmtime()))
    # if max_num_iteration > 0 and num_iteration >= max_num_iteration:
    #   break

if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('vis_logdir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
