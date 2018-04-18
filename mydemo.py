import os
import logging
import argparse
import numpy as np
import svgwrite

import drawing
import lyrics
from rnn import rnn
import pandas as pd

from wand.api import library
import wand.color
from wand.image import Image
import cv2
class Hand(object):

    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.nn = rnn(
            log_dir='logs',
            checkpoint_dir='checkpoints',
            prediction_dir='predictions',
            learning_rates=[.0001, .00005, .00002],
            batch_sizes=[32, 64, 64],
            patiences=[1500, 1000, 500],
            beta1_decays=[.9, .9, .9],
            validation_batch_size=32,
            optimizer='rms',
            num_training_steps=100000,
            warm_start_init_step=17900,
            regularization_constant=0.0,
            keep_prob=1.0,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=2000,
            log_interval=20,
            logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10
        )
        self.nn.restore()

    def write(self, filename, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None):
       # valid_char_set = set(['\x00', ' ', '/', '\', '_', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', 'Q', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
        valid_char_set = set(drawing.alphabet) 
        # print(drawing.alphabet)
       # {'L', 'T', 'b', 'g', 'S', '5', '7', ':', 'J', 'K', '"', 'd', 'Y', 'f', 'A', '-', 'm', 'x', '0', 't', 'o', '\x00', '3', 'w', 's', 'V', "'", 'N', '8', 'e', 'q', '1', 'a', 'O', 'i', 'l', 'B', '/', '_', ')', 'D', 'F', 'W', 'G', 'r', 'h', 'H', 'v', ' ', 'R', '2', 'M', '6', 'p', 'y', 'j', 'c', ';', 'I', 'U', 'E', 'k', '4', 'P', '?', 'u', ',', 'n', '(', '.', 'C', 'z', '9'}
        for line_num, line in enumerate(lines):
            if len(line) > 75:
                raise ValueError(
                    (
                        "Each line must be at most 75 characters. "
                        "Line {} contains {}"
                    ).format(line_num, len(line))
                )

            for char in line:
                if char not in valid_char_set:
                    raise ValueError(
                        (
                            "Invalid character {} detected in line {}. "
                            "Valid character set is {}"
                        ).format(char, line_num, valid_char_set)
                    )

        strokes = self._sample(lines, biases=biases, styles=styles)
        self._draw(strokes, lines, filename, line_num=line_num,stroke_colors=stroke_colors, stroke_widths=stroke_widths)

    def _sample(self, lines, biases=None, styles=None):
        num_samples = len(lines)
        max_tsteps = 40*max([len(i) for i in lines])
        biases = biases if biases is not None else [0.5]*num_samples

        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        if styles is not None:
            for i, (cs, style) in enumerate(zip(lines, styles)):
                x_p = np.load('styles/style-{}-strokes.npy'.format(style))
                c_p = np.load('styles/style-{}-chars.npy'.format(style)).tostring().decode('utf-8')

                c_p = str(c_p) + " " + cs
                c_p = drawing.encode_ascii(c_p)
                c_p = np.array(c_p)

                x_prime[i, :len(x_p), :] = x_p
                x_prime_len[i] = len(x_p)
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)

        else:
            for i in range(num_samples):
                encoded = drawing.encode_ascii(lines[i])
                chars[i, :len(encoded)] = encoded
                chars_len[i] = len(encoded)

        [samples] = self.nn.session.run(
            [self.nn.sampled_sequence],
            feed_dict={
                self.nn.prime: styles is not None,
                self.nn.x_prime: x_prime,
                self.nn.x_prime_len: x_prime_len,
                self.nn.num_samples: num_samples,
                self.nn.sample_tsteps: max_tsteps,
                self.nn.c: chars,
                self.nn.c_len: chars_len,
                self.nn.bias: biases
            }
        )
        samples = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
        return samples

    def _draw(self, strokes, lines, filename, line_num,stroke_colors=None, stroke_widths=None):
        stroke_colors = stroke_colors or ['black']*len(lines)
        stroke_widths = stroke_widths or [2]*len(lines)

        line_height = 80
        view_width = 500
        view_height = line_height*(len(strokes) + 1)

        dwg = svgwrite.Drawing(filename='p1.svg')
        dwg.viewbox(width=view_width, height=view_height)
        dwg.add(dwg.rect(insert=(0, 0), size=(view_width, view_height), fill='white'))

        initial_coord = np.array([0, -(3*line_height / 4)])
        for offsets, line, color, width in zip(strokes, lines, stroke_colors, stroke_widths):

            if not line:
                initial_coord[1] -= line_height
                continue

            offsets[:, :2] *= 1.2
            strokes = drawing.offsets_to_coords(offsets)
            strokes = drawing.denoise(strokes)
            strokes[:, :2] = drawing.align(strokes[:, :2])

            strokes[:, 1] *= -1
            strokes[:, :2] -= strokes[:, :2].min() + initial_coord
            strokes[:, 0] += (view_width - strokes[:, 0].max()) / 2

            prev_eos = 1.0
            p = "M{},{} ".format(0, 0)
            for x, y, eos in zip(*strokes.T):
                p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
                prev_eos = eos
            path = svgwrite.path.Path(p)
            path = path.stroke(color=color, width=width, linecap='round').fill("none")
            #print(path)
            dwg.add(path)

            initial_coord[1] -= line_height

            dwg.save()
        
        
            with Image(filename='p1.svg', format='svg') as img:
                img.format='png'
                img.save(filename="s.png")
            img_small = cv2.imread("s.png")
            img_large = cv2.imread("templates/large1.jpg")
            #print(img_large.shape)
            #overlay_image_alpha(img_large,
            #        img_small[:, :, :],
            #        (0, 0),
            #        img_small[:, :, :] / 255.0)

            s_img = cv2.imread("s.png",0)
           # print(s_img.shape)
            l_img = cv2.imread("large1.jpg",0)
           # print(l_img.shape) 
            x_offset=100  
            y_offset=200 + (line_num) * 75
            l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
            
            outfile = "output/" + filename


            cv2.imwrite(outfile,l_img)
            
            #try: 
            #    os.remove("s.png")
            #    os.remove("p1.png")
            #except: pass

            
def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    print(y1,y2,x1,x2)

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]
    print(channels)

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha
    print("large",img.shape)
    print("small",img_overlay.shape)

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Prescriptions Data")
    parser.add_argument("NumPres", help="Number of Prescriptions to Generate")
    args = parser.parse_args()
    n = int(args.NumPres)

    df = pd.read_csv('drug.csv')
    names = df['first_name'] + " " + df['last_name']
    medlines =  df['BNF NAME                                    '].astype(str)
    filenams = "Prescrition" +  df['Seq'].astype(int).astype(str) + ".png"
    hand = Hand()
    for idx,drugs in enumerate(medlines[:n]):
        lines = []
        #lines.append(names[idx])
        #lines.append(df['Address'][idx])
        lines.append(drugs[:40])
        lines.append(drugs[40:])
        #print(lines[0])
        #print(lines[1])
        biases = [.75 for i in lines]
        styles = [9 for i in lines]
        #Call write function to generate hanwritings.
        hand.write(
            filename=filenams[idx],
            lines=lines,
            biases=biases,
            styles=styles
        )
