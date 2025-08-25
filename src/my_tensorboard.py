from PIL import Image
from io import BytesIO
import tensorboardX as tb
from tensorboardX import SummaryWriter
from tensorboardX.summary import Summary
import numpy as np
import cv2

def process_img_for_logging(img, max_size=1024.):
    max_size *= 1.
    fact = img.shape[0] / max_size if img.shape[0] > max_size else 1.# Width to 512
    img = cv2.resize(img, (int(img.shape[1]/fact), int(img.shape[0]/fact)))
    return img

class TensorBoard(object):
    def __init__(self, model_dir):
        self.summary_writer = SummaryWriter(model_dir)
    def add_image(self, tag, img, step=None, max_size=1024.):
        ''' Expects channels last rgb image '''
        img = np.array(img)
        if max_size is not None:
            img = process_img_for_logging(img, max_size=max_size)

        if len(img.shape) == 2:
            img = Image.fromarray(img)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = np.array(img)
        img = np.transpose(img, (2, 0, 1))
        self.summary_writer.add_image(tag, img, step)
        self.summary_writer.flush()

    def add_scalar(self, tag, value, step=None):
        self.summary_writer.add_scalar(tag, value, step)

    def add_text(self, tag, text, step):
        self.summary_writer.add_text(tag, text, step)

    def add_figure(self, tag, fig, step):
        self.summary_writer.add_figure(tag, fig, step)


def extract_summary_from_tensorboardX(summary_path):
    """Extracts a summary from a tensorboardX Summary object."""
    if isinstance(summary, Summary):
        return summary.value
    elif isinstance(summary, list):
        return [extract_summary_from_tensorboardX(s) for s in summary]
    else:
        raise ValueError("Unsupported summary type: {}".format(type(summary)))


if __name__ == "__main__":
    # Example usage
    print(extract_summary_from_tensorboardX())