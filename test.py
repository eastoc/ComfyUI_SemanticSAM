from .semantic_sam import prepare_image, plot_multi_results, build_semantic_sam, SemanticSAMPredictor

original_image, input_image = prepare_image(image_pth='examples/dog.jpg')  # change the image path to your image
mask_generator = SemanticSAMPredictor(build_semantic_sam(model_type='L', ckpt='swinl_only_sam_many2many')) # model_type: 'L' / 'T', depends on your checkpint
iou_sort_masks, area_sort_masks = mask_generator.predict_masks(original_image, input_image, point='[[0.5, 0.5]]') # input point [[w, h]] relative location, i.e, [[0.5, 0.5]] is the center of the image
plot_multi_results(iou_sort_masks, area_sort_masks, original_image, save_path='../vis/')  # results and original images will be saved at save_path