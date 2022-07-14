from pred import instance_segmentation
seg_img = instance_segmentation()
seg_img.load_model("Models\mask_rcnn_model.001-2.181213.h5")
seg_img.segmentBatch(input_folder="test",show_bboxes=True,output_folder_name="out_test",box_thickness=1,text_size=0.1)