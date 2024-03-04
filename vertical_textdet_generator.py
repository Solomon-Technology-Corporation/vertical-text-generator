import os
import numpy as np
import cv2
import json
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import argparse

OPJ = os.path.join
NRR = np.random.randint


def load_text_image(numtext, len_text, textpath, text_list, use_mask=False):
    pick_test_list = []
    pick_test_w, pick_test_h = [], []
    for j in range(numtext):
        picktext = NRR(0, len_text)
        text_img = cv2.imread(OPJ(textpath, text_list[picktext]))
        if use_mask:
            tmask_img = cv2.imread(OPJ(textpath, text_mask_list[picktext]))#[...,None].repeat(3, 2)
            text_img = np.where(tmask_img > 0, text_img, 0)
        rands = np.random.uniform(0.7, 1.) if max(text_img.shape) > 800 else np.random.uniform(0.8, 1.)
        text_img = cv2.resize(text_img, None, fx=rands, fy=rands)
        h, w, _ = text_img.shape
        pick_test_w.append(w)
        pick_test_h.append(h)
        pick_test_list.append(text_img)
    return pick_test_w, pick_test_h, pick_test_list


def draw_text_images(bg_img, pick_test_list, pick_test_h, pick_test_w, limit_px, bg_h, bg_w):
    bg_img_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    pick_test_gt = []
    for j in range(len(pick_test_list)):
        h_t, w_t = pick_test_h[j], pick_test_w[j]
        text_img = pick_test_list[j]
        # print('text-{}, size: {} {}'.format(i, text_img.shape[:2], j))
        ctr = 0

        text_box = dict(transcription="ii")
        while True:
            try:
                ctr += 1
                '''define random position'''
                h_pos, w_pos = np.random.randint(limit_px, bg_h - limit_px - h_t), np.random.randint(limit_px, bg_w - limit_px - w_t)
                
                '''check if vertical text is in the image'''
                if bg_img_mask[h_pos, w_pos] == 0 and bg_img_mask[h_pos+h_t, w_pos+w_t] == 0:
                    bg_img_mask[h_pos:h_pos+h_t, w_pos:w_pos+w_t] = 1
                    bg_img[h_pos:h_pos+h_t, w_pos:w_pos+w_t] = np.where(text_img > 0, text_img, bg_img[h_pos:h_pos+h_t, w_pos:w_pos+w_t])
                    # bg_img[h_pos:h_pos+h_t, w_pos:w_pos+w_t] = np.where(text_img > 0, text_img*0.8 + bg_img[h_pos:h_pos+h_t, w_pos:w_pos+w_t]*0.2, bg_img[h_pos:h_pos+h_t, w_pos:w_pos+w_t])
                    text_box["points"] = [[w_pos, h_pos], [w_pos+w_t, h_pos], [w_pos+w_t, h_pos+h_t], [w_pos, h_pos+h_t]]
                    pick_test_gt.append(text_box)
                    break
                
                '''if after 50 loop, we still can't find empty positions from defined random position, we will skip this text'''
                if ctr == 50:
                    ctr = -1
                    break
            except:
                # print(i, '>>> ERROR random pos: ', limit_px, bg_h, bg_w, h_t, w_t)
                if ctr == 50:
                    ctr = -1
                    break
                continue
    return pick_test_gt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Vertical-Text-Generator',
                    description='What the program does',
                    epilog='')
    parser.add_argument('-um', '--use_mask', action="store_true")
    parser.add_argument('-text', '--textpath', type=str, default='out/')
    parser.add_argument('-bg', '--bgpath', type=str, default='bg_img/')
    parser.add_argument('-opath', '--outpath', type=str, default='vtg_out/')
    parser.add_argument('-opl', '--outpath_label', type=str, default='vtg_out')
    parser.add_argument('-n', '--num_sample', type=int, default=100)
    parser.add_argument('-lp', '--limit_px', type=int, default=2)
    args = vars(parser.parse_args())
    
    textpath = args['textpath']
    bgpath = args['bgpath']
    resultpath = args['outpath']
    os.makedirs(resultpath, exist_ok=True)
    text_list = [tp for tp in os.listdir(textpath) if tp.endswith('.jpg')]
    text_mask_list = [tp.replace('.jpg', '_mask.png') for tp in text_list]
    text_label_list = [tp.replace('.jpg', '_boxes.txt') for tp in text_list]
    bg_list = os.listdir(bgpath)
    len_text = len(text_list)
    len_bg = len(bg_list)
    print('[VTG] len text: ', len_text)
    print('[VTG] check text value: ', text_list[0], text_mask_list[0])
    print('[VTG] len background: ', len_bg)

    pick_test_gt_list = []
    limit_px = args['limit_px']
    gen_images = args['num_sample']

    for i in range(gen_images):
        pick_bg = NRR(0, len_bg)
        bg_img = cv2.imread(OPJ(bgpath, bg_list[pick_bg]))
        bg_h, bg_w = bg_img.shape[:2]
        bg_min = min(bg_h, bg_w)
        imname = f'{resultpath}image{i}.jpg'
        
        numtext = NRR(3, 7 if bg_min < 300 else 10 if bg_min < 1000 else 15)
        pick_test_w, pick_test_h, pick_test_list = load_text_image(numtext, len_text, textpath, text_list, use_mask=args['use_mask'])
        
        hmax, wmax = max(pick_test_h), max(pick_test_w)
        pick_test_gt = draw_text_images(bg_img, pick_test_list, pick_test_h, pick_test_w, limit_px, bg_h, bg_w)
        
        if len(pick_test_gt) > 0:
            pick_test_gt = json.dumps(pick_test_gt)
            pick_test_gt_list.append(imname + '\t' + str(pick_test_gt) + '\r\n')
        
        cv2.imwrite(imname, bg_img)
        print(f'Generate {i}/{gen_images} done.', len(pick_test_gt_list))

    with open('label_{}.txt'.format(args['outpath_label']), 'w', encoding='utf-8') as wt:
        wt.writelines(pick_test_gt_list)
