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
class HWPrescription(object):

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
        valid_char_set = set(drawing.alphabet) 
        #print(valid_char_set)
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
        stroke_colors = stroke_colors or ['black']*4
        stroke_widths = stroke_widths or [2]*4
        i=1
        #i+=1            
        img_large = cv2.imread("templates/large1.jpg")
        l_img = cv2.imread("large1.jpg",0)
        for offsets, line, color, width in zip(strokes, lines, stroke_colors, stroke_widths):
            
            svgfil = 'p' + str(i) + '.svg'
            
            if i == 1:
                line_height = 30
                view_width = 200                
                dwg1 = svgwrite.Drawing(filename=svgfil)
                dwg = dwg1
            

            if i == 2:
                line_height = 30
                view_width = 300  
                dwg2 = svgwrite.Drawing(filename=svgfil)
                dwg = dwg2

            if i == 3:
                line_height = 60
                view_width = 450   
                dwg3 = svgwrite.Drawing(filename=svgfil)
                dwg = dwg3
                
            if i == 4:
                line_height = 60
                view_width = 450
                dwg4 = svgwrite.Drawing(filename=svgfil)
                dwg = dwg4
    
                        
            #line_height = 50
            #view_width = 300
            
            view_height = line_height*2
            
    
            dwg.viewbox(width=view_width, height=view_height)
            dwg.add(dwg.rect(insert=(0, 0), size=(view_width, view_height), fill='white'))
            initial_coord = np.array([0, -(3*line_height / 4)])
            
            if not line:
                initial_coord[1] -= line_height
                continue
            
            offsets[:, :2] *= 1.1
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
            if i==1:
                path1 = svgwrite.path.Path(p)
                path1 = path1.stroke(color=color, width=width, linecap='round').fill("none")
                dwg.add(path1)
            if i==2:
                path2 = svgwrite.path.Path(p)
                path2 = path2.stroke(color=color, width=width, linecap='round').fill("none")
                dwg.add(path2)
            if i==3:
                path3 = svgwrite.path.Path(p)
                path3 = path3.stroke(color=color, width=width, linecap='round').fill("none")
                dwg.add(path3)
            if i==4:
                path4 = svgwrite.path.Path(p)
                path4 = path4.stroke(color=color, width=width, linecap='round').fill("none")
                dwg.add(path4)

            #initial_coord[1] -= line_height

            dwg.save()
            with Image(filename=svgfil, format='svg') as img:
            
                img.format='png'
                img.save(filename="s.png")


            s_img = cv2.imread("s.png",0)

            #control placement of text on the form
            if i == 1: 
                x_offset=135  
                y_offset=50
            if i==2:
                x_offset=155
                y_offset=110
            if i==3:
                x_offset=150
                y_offset=300
            if i==4:
                x_offset=150
                y_offset=390
            
                
            
            l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
            
            

            
            outfile = "output/" + filename
            i=i+1


            cv2.imwrite(outfile,l_img)
            s_img  = np.ones(s_img.shape) *int(255)
            cv2.imwrite("s.png",s_img)

            
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
    medlines.replace('_',' ',regex=True,inplace=True)
    medlines.replace('Z','z',regex=True,inplace=True)
    medlines.replace('X','x',regex=True,inplace=True)
    medlines.replace('Q','q',regex=True,inplace=True)
    medlines.replace('%',' ',regex=True,inplace=True)
    medlines.replace('#',' ',regex=True,inplace=True)
    medlines.replace('@',' ',regex=True,inplace=True)
    medlines.replace('&',' ',regex=True,inplace=True)

    
    filenams = "Prescription" +  df['Seq'].astype(int).astype(str) + ".png"
    hand = Hand()
    for idx,drugs in enumerate(medlines[:n]):
        lines = []
        lines.append(names[idx])
        lines.append(df['Address'][idx])
        lines.append(drugs[:40])
        lines.append(drugs[40:])
        #print(lines[0])
        #print(lines[1])
        biases = [df['Biases'][idx] for i in lines]
        styles = [df['Styles'][idx] for i in lines]
        #Call write function to generate hanwritings.
        hand.write(
            filename=filenams[idx],
            lines=lines,
            biases=biases,
            styles=styles
        )
