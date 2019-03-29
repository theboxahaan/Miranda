import glob
import os
import random
import sys
import random
import math
import json
from collections import defaultdict
import time

import cv2
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage.filters import rank_filter


def dilate(ary, N, iterations): 
    """Dilate using an NxN '+' sign shape. ary is np.uint8."""
    
    kernel = np.zeros((N,N), dtype=np.uint8)
    kernel[(N-1)//2,:] = 1  # Bug solved with // (integer division)
    
    dilated_image = cv2.dilate(ary / 255, kernel, iterations=iterations)
    
    kernel = np.zeros((N,N), dtype=np.uint8)
    kernel[:,(N-1)//2] = 1  # Bug solved with // (integer division)
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)
    return dilated_image


def props_for_contours(contours, ary):
    """Calculate bounding box & the number of set pixels for each contour."""
    c_info = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        c_im = np.zeros(ary.shape)
        cv2.drawContours(c_im, [c], 0, 255, -1)
        c_info.append({
            'x1': x,
            'y1': y,
            'x2': x + w - 1,
            'y2': y + h - 1,
            'sum': np.sum(ary * (c_im > 0))/255
        })
    return c_info


def union_crops(crop1, crop2):
    """Union two (x1, y1, x2, y2) rects."""
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return min(x11, x12), min(y11, y12), max(x21, x22), max(y21, y22)


def intersect_crops(crop1, crop2):
    x11, y11, x21, y21 = crop1
    x12, y12, x22, y22 = crop2
    return max(x11, x12), max(y11, y12), min(x21, x22), min(y21, y22)


def crop_area(crop):
    x1, y1, x2, y2 = crop
    return max(0, x2 - x1) * max(0, y2 - y1)


def find_border_components(contours, ary):
    borders = []
    area = ary.shape[0] * ary.shape[1]
    for i, c in enumerate(contours):
        x,y,w,h = cv2.boundingRect(c)
        if w * h > 0.5 * area:
            borders.append((i, x, y, x + w - 1, y + h - 1))
    return borders


def angle_from_right(deg):
    return min(deg % 90, 90 - (deg % 90))


def remove_border(contour, ary):
    """Remove everything outside a border contour."""
    # Use a rotated rectangle (should be a good approximation of a border).
    # If it's far from a right angle, it's probably two sides of a border and
    # we should use the bounding box instead.
    c_im = np.zeros(ary.shape)
    r = cv2.minAreaRect(contour)
    degs = r[2]
    if angle_from_right(degs) <= 10.0:
        box = cv2.boxPoints(r)
        box = np.int0(box)
        cv2.drawContours(c_im, [box], 0, 255, -1)
        cv2.drawContours(c_im, [box], 0, 0, 4)
    else:
        x1, y1, x2, y2 = cv2.boundingRect(contour)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 255, -1)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 0, 4)

    return np.minimum(c_im, ary)


def find_components(edges, max_components=16):
    """Dilate the image until there are just a few connected components.
    Returns contours for these components."""
    # Perform increasingly aggressive dilation until there are just a few
    # connected components.
    
    count = 21
    dilation = 5
    n = 100
    while count > 16:
        n += 1
        dilated_image = dilate(edges, N=3, iterations=n)
        dilated_image = np.uint8(dilated_image)
        _, contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)
    
##    print(dilation, n)
    #Image.fromarray(edges).show()
    #Image.fromarray(255 * dilated_image).show()
    return contours


def find_optimal_components_subset(contours, edges):
    """Find a crop which strikes a good balance of coverage/compactness.
    Returns an (x1, y1, x2, y2) tuple.
    """
    c_info = props_for_contours(contours, edges)
    c_info.sort(key=lambda x: -x['sum'])
    total = np.sum(edges) / 255
    area = edges.shape[0] * edges.shape[1]

    c = c_info[0]
    del c_info[0]
    this_crop = c['x1'], c['y1'], c['x2'], c['y2']
    crop = this_crop
    covered_sum = c['sum']

    while covered_sum < total:
        changed = False
        recall = 1.0 * covered_sum / total
        prec = 1 - 1.0 * crop_area(crop) / area
        f1 = 2 * (prec * recall / (prec + recall))
        #print '----'
        for i, c in enumerate(c_info):
            this_crop = c['x1'], c['y1'], c['x2'], c['y2']
            new_crop = union_crops(crop, this_crop)
            new_sum = covered_sum + c['sum']
            new_recall = 1.0 * new_sum / total
            new_prec = 1 - 1.0 * crop_area(new_crop) / area
            new_f1 = 2 * new_prec * new_recall / (new_prec + new_recall)

            # Add this crop if it improves f1 score,
            # _or_ it adds 25% of the remaining pixels for <15% crop expansion.
            # ^^^ very ad-hoc! make this smoother
            remaining_frac = c['sum'] / (total - covered_sum)
            new_area_frac = 1.0 * crop_area(new_crop) / crop_area(crop) - 1
            if new_f1 > f1 or (
                    remaining_frac > 0.25 and new_area_frac < 0.15):
##                print('%d %s -> %s / %s (%s), %s -> %s / %s (%s), %s -> %s' % (
##                        i, covered_sum, new_sum, total, remaining_frac,
##                        crop_area(crop), crop_area(new_crop), area, new_area_frac,
##                        f1, new_f1))
                crop = new_crop
                covered_sum = new_sum
                del c_info[i]
                changed = True
                break

        if not changed:
            break

    return crop


def pad_crop(crop, contours, edges, border_contour, pad_px=15):
    """Slightly expand the crop to get full contours.
    This will expand to include any contours it currently intersects, but will
    not expand past a border.
    """
    bx1, by1, bx2, by2 = 0, 0, edges.shape[0], edges.shape[1]
    if border_contour is not None and len(border_contour) > 0:
        c = props_for_contours([border_contour], edges)[0]
        bx1, by1, bx2, by2 = c['x1'] + 5, c['y1'] + 5, c['x2'] - 5, c['y2'] - 5

    def crop_in_border(crop):
        x1, y1, x2, y2 = crop
        x1 = max(x1 - pad_px, bx1)
        y1 = max(y1 - pad_px, by1)
        x2 = min(x2 + pad_px, bx2) 
        y2 = min(y2 + pad_px, by2)
        return crop
    
    crop = crop_in_border(crop)

    c_info = props_for_contours(contours, edges)
    changed = False
    for c in c_info:
        this_crop = c['x1'], c['y1'], c['x2'], c['y2']
        this_area = crop_area(this_crop)
        int_area = crop_area(intersect_crops(crop, this_crop))
        new_crop = crop_in_border(union_crops(crop, this_crop))
        if 0 < int_area < this_area and crop != new_crop:
##            print('%s -> %s' % (str(crop), str(new_crop)))
            changed = True
            crop = new_crop

    if changed:
        return pad_crop(crop, contours, edges, border_contour, pad_px)
    else:
        return crop


def downscale_image(im, max_dim=2048):
    """Shrink im until its longest dimension is <= max_dim.
    Returns new_image, scale (where scale <= 1).
    """
    a, b = im.size
    if max(a, b) <= max_dim:
        return 1.0, im

    scale = 1.0 * max_dim / max(a, b)
    new_im = im.resize((int(a * scale), int(b * scale)), Image.ANTIALIAS)
    return scale, new_im


def process_image(path, out_path):

    orig_im = Image.open(path)
    scale, im = downscale_image(orig_im)

    edges = cv2.Canny(np.asarray(im), 100, 200)

    # TODO: dilate image _before_ finding a border. This is crazy sensitive!
    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    borders = find_border_components(contours, edges)
    borders.sort(key=lambda i_x1_y1_x2_y2: (i_x1_y1_x2_y2[3] - i_x1_y1_x2_y2[1]) * (i_x1_y1_x2_y2[4] - i_x1_y1_x2_y2[2]))

    border_contour = None
    if len(borders):
        border_contour = contours[borders[0][0]]
        edges = remove_border(border_contour, edges)

    edges = 255 * (edges > 0).astype(np.uint8)

    # Remove ~1px borders using a rank filter.
    maxed_rows = rank_filter(edges, -4, size=(1, 20))
    maxed_cols = rank_filter(edges, -4, size=(20, 1))
    debordered = np.minimum(np.minimum(edges, maxed_rows), maxed_cols)
    edges = debordered
    contours_start = time.time()
    contours = find_components(edges)
    print("find_components -> ", time.time() - contours_start)
    if len(contours) == 0:
        print('%s -> (no text!)' % path)
        return

    
    find_optimal_components_subset_start = time.time()
    crop = find_optimal_components_subset(contours, edges)
    print("find_optimal_components_subset -> ", time.time() - find_optimal_components_subset_start)

    pad_crop_start = time.time()
    crop = pad_crop(crop, contours, edges, border_contour)
    print("pad_crop - > ", time.time() - pad_crop_start)
    crop = [int(x / scale) for x in crop]  # upscale to the original image size.
    crop[0] -= 100
    crop[1] -= 100
    crop[2] += 100
    crop[3] += 100
    text_im = orig_im.crop(crop)
    text_im = np.array(text_im)
    imgSize = np.shape(text_im)
    new_mask = np.zeros(imgSize, dtype="uint8")
    
    final = cv2.cvtColor(text_im, cv2.COLOR_RGB2GRAY)
    # Adaptive Thresholding requires the blocksize to be odd and bigger than 1
    blockSize = 1 // 8 * imgSize[0] // 2 * 2 + 1
    if blockSize <= 1:
        blockSize = imgSize[0] // 2 * 2 + 1
    const = 10

    mask = cv2.adaptiveThreshold(final, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = blockSize, C = const)
    rmask = cv2.bitwise_not(mask)

    text_im = cv2.bitwise_and(text_im, text_im, mask=rmask)

    text_im[np.where((text_im!=[0,0,0]).all(axis=2))] = [205, 171, 33]
    cv2.imwrite(out_path, text_im)
    print('%s -> %s' % (path, out_path))


##def brightness():
##
##    img = cv2.imread('11.jpg')
##    print(img.shape)
####    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
####
####
####    imghsv[:,:,2] = [[max(pixel - 25, 0) if pixel < 190 else min(pixel + 25, 255) for pixel in row] for row in imghsv[:,:,2]]
##    bright = np.ones(img.shape, np.uint8)*50
##
##    img += bright
##    cv2.imwrite('test.jpg', img)
##
####    return cur_img
####    print("Where")
####    img = cv2.imread('9.jpg')
####    print("are")
####    rgb_planes = cv2.split(img)
####
####    result_planes = []
####    result_norm_planes = []
####    for plane in rgb_planes:
####        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
####        bg_img = cv2.medianBlur(dilated_img, 21)
####        diff_img = 255 - cv2.absdiff(plane, bg_img)
####        #norm_img = cv2.normalize(diff_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
####        print("here")
####        result_planes.append(diff_img)
####        #result_norm_planes.append(norm_img)
####        
######    print("here")
####    result = cv2.merge(result_planes)
####    #result_norm = cv2.merge(result_norm_planes)
####    print("yu")
####    result = cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)
####    cv2.imwrite('shadows_out.jpg', result)
####    #cv2.imwrite('shadows_out_norm.jpg', result_norm)
####    print("Kif")
####    #-----Reading the image-----------------------------------------------------
####    img = cv2.imread('11.jpg')
####    print(img.shape)
######    cv2.imshow("img",img) 
####
####    #-----Converting image to LAB Color model----------------------------------- 
####    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
######    cv2.imshow("lab",lab)
####
####    #-----Splitting the LAB image to different channels-------------------------
####    l, a, b = cv2.split(lab)
######    cv2.imshow('l_channel', l)
######    cv2.imshow('a_channel', a)
######    cv2.imshow('b_channel', b)
####
####    #-----Applying CLAHE to L-channel-------------------------------------------
####    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
####    cl = clahe.apply(l)
######    cv2.imshow('CLAHE output', cl)
####
####    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
####    limg = cv2.merge((cl,a,b))
######    cv2.imshow('limg', limg)
####
####    #-----Converting image from LAB Color model to RGB model--------------------
####    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
######    cv2.imshow('final', final)
####    cv2.imwrite('shadows_out.jpg', final)
####
####    #_____END_____#    


def brightness():

    img = cv2.imread("11.jpg")

    mean = img.mean()
    print(mean)



def remove_background(i):
	img = Image.open(i)
	img = img.convert("RGBA")
	datas = img.getdata()

	newData = []
	for item in datas:
	    if item[0] == 255 and item[1] == 255 and item[2] == 255:
	        newData.append((255, 255, 255, 0))
	    else:
	        if item[0] > 150:
	            newData.append((0, 0, 0, 255))
	        else:
	            newData.append(item)
	            #print(item)


	img.putdata(newData)
	img.save(i.split(".")[0]+".png", "PNG")
	print(i)
	print("Saved")


if __name__ == '__main__':
    # List the files for which Signature is to be detected
    files = ["11.jpg"]#, "12.jpg", "13.jpg", "14.jpg"]
    for path in files:
        out_path = path.replace('.jpg', '_sig_detected.jpg')
        #out_path = path.replace('.png', '.crop.png')  # .png as input
        if os.path.exists(out_path): continue
        try:
##            start = time.time()
##            process_image(path, out_path)
##            print(time.time() - start)
##            remove_background(out_path)
            brightness()
        except Exception as e:
            print('%s %s' % (path, e))
