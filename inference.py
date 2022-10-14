import pytorch_lightning as pl
from argparse import ArgumentParser
from src.utils.config import ConfigParser
import src.dataloaders as dataloader_module
import src.models as model_module
from pytorch_lightning import loggers as pl_loggers
import warnings, cv2, torch
import numpy as np
from glob import glob

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument('--config', default = './config/mnist.json', type = str)
    parser.add_argument('--model_checkpoint', required=True, type = str)
    parser.add_argument('--data_dir', required=True, type = str)
    config = ConfigParser.from_args(parser)
    
    args = parser.parse_args()
    model = config.init_obj('model', model_module)
    model = model.load_from_checkpoint(args.model_checkpoint)
    model.eval()

    test_images = glob("{}/*jpg".format(args.data_dir))
    print(f"Test instances: {len(test_images)}")
    
    CLASSES = config['dataloader']['args']['classes']
    detection_threshold = 0.8

    for i in range(len(test_images)):
        # get the image file name for saving output later on
        image_name = test_images[i].split('/')[-1].split('.')[0]
        image = cv2.imread(test_images[i])
        orig_image = image.copy()
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image)
        
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
            
            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 2)
                cv2.putText(orig_image, pred_classes[j], 
                            (int(box[0]), int(box[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                            2, lineType=cv2.LINE_AA)
            cv2.imwrite(f"./test_predictions/{image_name}.jpg", orig_image,)
        print(f"Image {i+1} done...")
        print('-'*50)
    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()