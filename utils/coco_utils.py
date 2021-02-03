import os
from pathlib import Path
from pprint import pprint

import torch
from torchvision import transforms
from torchvision.datasets import CocoDetection
from matplotlib import patches, pyplot as plt
import lightnet as ln
from torch.utils.data.dataloader import DataLoader
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config import PROJECT_ROOT_DIR

coco_num_classes = 80
coco_anchors_simple = [(0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434), (7.88282, 3.52778), (9.77052, 9.16828)]

coco_old_category_id_to_new_id = {1: 0,
 2: 1,
 3: 2,
 4: 3,
 5: 4,
 6: 5,
 7: 6,
 8: 7,
 9: 8,
 10: 9,
 11: 10,
 13: 11,
 14: 12,
 15: 13,
 16: 14,
 17: 15,
 18: 16,
 19: 17,
 20: 18,
 21: 19,
 22: 20,
 23: 21,
 24: 22,
 25: 23,
 27: 24,
 28: 25,
 31: 26,
 32: 27,
 33: 28,
 34: 29,
 35: 30,
 36: 31,
 37: 32,
 38: 33,
 39: 34,
 40: 35,
 41: 36,
 42: 37,
 43: 38,
 44: 39,
 46: 40,
 47: 41,
 48: 42,
 49: 43,
 50: 44,
 51: 45,
 52: 46,
 53: 47,
 54: 48,
 55: 49,
 56: 50,
 57: 51,
 58: 52,
 59: 53,
 60: 54,
 61: 55,
 62: 56,
 63: 57,
 64: 58,
 65: 59,
 67: 60,
 70: 61,
 72: 62,
 73: 63,
 74: 64,
 75: 65,
 76: 66,
 77: 67,
 78: 68,
 79: 69,
 80: 70,
 81: 71,
 82: 72,
 84: 73,
 85: 74,
 86: 75,
 87: 76,
 88: 77,
 89: 78,
 90: 79}

coco_labels_list = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "street sign",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "hat",
  "backpack",
  "umbrella",
  "shoe",
  "eye glasses",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "plate",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "mirror",
  "dining table",
  "window",
  "desk",
  "toilet",
  "door",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "blender",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
  "hair brush"
]
coco_labels_dict = {}
for i, label in enumerate(coco_labels_list):
  i += 1
  if i not in coco_old_category_id_to_new_id:
    continue
  coco_labels_dict[coco_old_category_id_to_new_id[i]] = label

def convert_cat_id(cat):
  cat = int(cat)
  if 0 <= cat <= 10:
    cat = cat + 1
  elif 11 <= cat <= 23:
    cat = cat + 2
  elif 24 <= cat <= 25:
    cat = cat + 3
  elif 26 <= cat <= 39:
    cat = cat + 5
  elif 40 <= cat <= 59:
    cat = cat + 6
  elif cat == 60:
    cat = cat + 7
  elif cat == 61:
    cat = cat + 9
  elif 62 <= cat <= 72:
    cat = cat + 10
  elif 73 <= cat <= 79:
    cat = cat + 11
  else:
    raise Exception('Category ID not in range')
  return cat

def transform_coco_datum(image, annotations_in_image):
  """
  Function for transforming each datum in the COCO dataset for input in training / validation
  """
  original_image_width, original_image_height = image.size
  _image_transformer = transforms.Compose([
    transforms.Resize((416, 416), interpolation=2),
    transforms.ToTensor(),
  ])

  def _train_target_transformer(annotations_in_image):
    """
    """
    def _annotation_transformer(annotation):
      """
      Helper function for transforming each annotation
      """
      _scale_value = lambda value_to_scale, original_dimension: round(value_to_scale / original_dimension, 3)

      original_bbox_x, original_bbox_y, original_bbox_width, original_bbox_height = annotation['bbox']
      scaled_bbox_x = _scale_value(original_bbox_x, original_image_width) 
      scaled_bbox_y = _scale_value(original_bbox_y, original_image_height) 
      scaled_bbox_w = _scale_value(original_bbox_width, original_image_width) 
      scaled_bbox_h = _scale_value(original_bbox_height, original_image_height)

      # lightnet expects x and y to be in the center
      scaled_bbox_x += scaled_bbox_w / 2
      scaled_bbox_y += scaled_bbox_h / 2
      new_category_id = coco_old_category_id_to_new_id[annotation['category_id']]

      scaled_bbox_tensor = torch.Tensor([new_category_id, scaled_bbox_x, scaled_bbox_y, scaled_bbox_w, scaled_bbox_h])

      return {
          'image_id': annotation['image_id'],
          'original_bbox': annotation['bbox'],
          'scaled_bbox': scaled_bbox_tensor,
          'category_id': new_category_id,
          'text_label': coco_labels_dict[new_category_id]
      }

    transformed_annotations = []
    scaled_bboxes = torch.zeros((100, 5))
    scaled_bboxes[..., 0] = -1
    for i, original_annotation in enumerate(annotations_in_image):
      transformed_annotation = _annotation_transformer(original_annotation)
      transformed_annotations.append(transformed_annotation)
      scaled_bboxes[i] = transformed_annotation['scaled_bbox']

    return {
        'annotations': transformed_annotations,
        'scaled_bboxes': scaled_bboxes
    }

  return _image_transformer(image), _train_target_transformer(annotations_in_image)

def show_image_predictions(image_tensor, predicted_bramboxes, true_bramboxes=None, output_image_name='sample.png'):
  def draw_brambox(brambox, edgecolor='r', facecolor='red'):
    x = brambox['x_top_left']
    y = brambox['y_top_left']
    width = brambox['width']
    height = brambox['height']
    category_text = brambox['class_label']
    if brambox['class_label'].isnumeric():
      category_text = coco_labels_dict[int(brambox['class_label'])]
    confidence = brambox['confidence']

    # add text
    box_text = f'{category_text} = {confidence}'
    plt.text(x, y, box_text, bbox=dict(facecolor=facecolor, alpha=0.5))

    # Create a Rectangle patch
    rect = patches.Rectangle((x, y), width, height,linewidth=1,edgecolor=edgecolor,facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

  if len(image_tensor.size()) == 4 and image_tensor.size()[0] == 1:
    image_tensor = image_tensor.squeeze(dim=0)

  # Create figure and axes
  fig, ax = plt.subplots(1)

  if true_bramboxes is not None:
    for i in range(len(true_bramboxes)):
      draw_brambox(true_bramboxes.iloc[i], 'g', 'green')

  for i in range(len(predicted_bramboxes)):
    draw_brambox(predicted_bramboxes.iloc[i], 'r', 'red')

  # Display the image
  ax.imshow(image_tensor.permute(1, 2, 0).detach().cpu())
  fig.savefig(os.path.join(PROJECT_ROOT_DIR, 'sample_outputs', output_image_name))
  
def collate_fn(items):
  images = torch.Tensor()
  scaled_bboxes = torch.Tensor()
  annotations = []
  for item in items:
    image, image_truths = item
    images = torch.cat((images, image.unsqueeze(0)))
    scaled_bboxes = torch.cat((scaled_bboxes, image_truths['scaled_bboxes'].unsqueeze(0)))
    annotations.append(image_truths['annotations'])
  return {
      'images': images,
      'scaled_bboxes': scaled_bboxes,
      'annotations': annotations
  }

def convert_gt_bboxes_to_prediction_bboxes(gt_bboxes):
  if len(gt_bboxes.size()) != 2:
    raise Exception('2D (num_boxes, 5) Tensor expected as gt_bboxes')
  num_bboxes = gt_bboxes.size()[0]
  prediction_bboxes = torch.zeros((num_bboxes, 7))
  prediction_bboxes[..., 0] = 0 # batch num
  prediction_bboxes[..., 1] = gt_bboxes[..., 1]
  prediction_bboxes[..., 2] = gt_bboxes[..., 2]
  prediction_bboxes[..., 3] = gt_bboxes[..., 3]
  prediction_bboxes[..., 4] = gt_bboxes[..., 4]
  prediction_bboxes[..., 5] = 1 # conf
  prediction_bboxes[..., 6] = gt_bboxes[..., 0]

  return prediction_bboxes

def convert_bramboxes_to_coco(image_id, bramboxes, original_height, original_width, resized_height=416, resized_width=416):
  _scale_value = lambda value_to_scale, original_dim, resized_dim: round(value_to_scale / resized_dim * original_dim, 1)

  coco_bboxes = []
  for i in range(len(bramboxes)):
    brambox = bramboxes.iloc[i]
    x = _scale_value(brambox['x_top_left'], original_width, resized_width)
    y = _scale_value(brambox['y_top_left'], original_height, resized_height)
    width = _scale_value(brambox['width'], original_width, resized_width)
    height = _scale_value(brambox['height'], original_height, resized_height)
    category_id = convert_cat_id(brambox['class_label'])
    confidence = round(brambox['confidence'], 3)
    coco_bboxes.append({
        'image_id': image_id,
        'category_id': category_id,
        'bbox': [x, y, width, height],
        'score': confidence
    })
  return coco_bboxes

def evaluate_coco(pred, gt):
  """
  example of gt: 'COCO2017/annotations/instances_val2017.json'
  """
  with open('out.json', 'w') as outfile:
    json.dump(pred, outfile)

  cocoGt = COCO(gt)
  cocoDt = cocoGt.loadRes('out.json')

  cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
  cocoEval.evaluate()
  cocoEval.accumulate()
  cocoEval.summarize()
  
if __name__ == '__main__':
  # Ensure that each operation is the reverse of another
  class_label_id = 59
  class_label_text = coco_labels_list[class_label_id - 1]
  assert class_label_text == 'pizza', 'Not pizza, was: {}'.format(class_label_text)

  class_index_from_model = coco_old_category_id_to_new_id[class_label_id]
  assert coco_labels_dict[class_index_from_model] == 'pizza', 'Not pizza, was: ' + coco_labels_dict[class_index_from_model]

  # Ensure that data can be loaded using a CocoDataset and a Dataloader
  coco_datum_transformer = transform_coco_datum_factory((416, 416), coco_labels_dict, torch.device('cpu'))
  coco_root_dir = Path(r'D:\FYPCode\COCO2017')
  coco_val_dataset = CocoDetection(os.path.join(coco_root_dir, 'images', 'val2017'), os.path.join(coco_root_dir, 'annotations', 'instances_val2017.json'), transforms = coco_datum_transformer)

  collate_fn = collate_fn_factory(torch.device('cpu'))
  sample_dataloader = DataLoader(coco_val_dataset, batch_size=1)
  sample_dataloader.collate_fn = collate_fn

  # And also ensure that the prediction bboxes are accurate
  for batch in sample_dataloader:
    images_in_batch, scaled_bboxes_in_batch = batch['images'], batch['scaled_bboxes']
    image, gt_bboxes_tensor = images_in_batch[0], scaled_bboxes_in_batch[0]
    bboxes_to_bramboxes = ln.data.transform.TensorToBrambox((416, 416))
    fake_model_output_bboxes = convert_gt_bboxes_to_prediction_bboxes(gt_bboxes_tensor)
    image_bramboxes = bboxes_to_bramboxes(fake_model_output_bboxes)
    show_image_predictions(image, [], image_bramboxes)
    print('Saved sample prediction output within image as sample.png')
    break