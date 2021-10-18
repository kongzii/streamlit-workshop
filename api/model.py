import os
import io
import json
import pickle
import numpy as np
from operator import itemgetter

import face_recognition

from PIL import Image

import typing as t

import tempfile

import torch
import clip
import cv2
from heapq import nsmallest
from tqdm import tqdm

FASHION_MODELS_FACES_PATH = "files/fashion_sales_assistant/Running_Examples/best_matches_faces"
FASHION_MODELS_PATH = "files/fashion_sales_assistant/Running_Examples/best_matches_models"

IMAGE_EXTENSIONS = ["bmp", "png", "jpg", "jpeg"]

TOP_N = 5

ESHOP_PATH = "files/fashion_sales_assistant/Running_Examples/eshop_database"


def get_resized_img(path: str):
    im = cv2.imread(path)
    scale = 320 / im.shape[0]
    return cv2.resize(
        im,
        (int(im.shape[1] * scale), int(im.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )


def get_cloth_from_model_image(
    model, preprocess, img_file_path, tops_list, bottoms_list, colors
):
    im = cv2.imread(img_file_path)
    scale = 320 / im.shape[0]
    # cv2_imshow(cv2.resize(im, (int(im.shape[1]*scale), int(im.shape[0]*scale)), interpolation = cv2.INTER_AREA))

    image = preprocess(Image.open(img_file_path)).unsqueeze(0).to(device)

    top_type_label_list = (
        ["a man in a {}.".format(top) for top in tops_list]
        + ["a man wearing a {}.".format(top) for top in tops_list]
        + ["a woman in a {}.".format(top) for top in tops_list]
        + ["a woman wearing a {}.".format(top) for top in tops_list]
        + ["a girl in a {}.".format(top) for top in tops_list]
        + ["a girl wearing a {}.".format(top) for top in tops_list]
    )
    top_type_label_list = ["a photo of " + p for p in top_type_label_list]
    text = clip.tokenize(top_type_label_list).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    top_type = tops_list[np.argmax(probs) % len(tops_list)]
    score = probs[np.argmax(probs)]

    res1 = list(zip(top_type_label_list, probs))
    res1 = list(sorted(res1, key=lambda tup: tup[1], reverse=True))
    # print(res1)

    top_color_label_list = [
        "a photo of a man wearing a {} {}.".format(
            color, tops_list[np.argmax(probs) % len(tops_list)]
        )
        for color in colors
    ] + [
        "a photo of a woman wearing a {} {}.".format(
            color, tops_list[np.argmax(probs) % len(tops_list)]
        )
        for color in colors
    ]
    text = clip.tokenize(top_color_label_list).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    top_color = colors[np.argmax(probs) % len(colors)]
    score = probs[np.argmax(probs)]

    res2 = list(zip(top_color_label_list, probs))
    res2 = list(sorted(res2, key=lambda tup: tup[1], reverse=True))
    # print(res2)

    bottom_type_label_list = (
        ["a man in {}.".format(bottom) for bottom in bottoms_list]
        + ["a man wearing {}.".format(bottom) for bottom in bottoms_list]
        + ["a woman in {}.".format(bottom) for bottom in bottoms_list]
        + ["a woman wearing {}.".format(bottom) for bottom in bottoms_list]
        + ["a girl in {}.".format(bottom) for bottom in bottoms_list]
        + ["a girl wearing {}.".format(bottom) for bottom in bottoms_list]
    )
    bottom_type_label_list = ["a photo of " + p for p in bottom_type_label_list]
    text = clip.tokenize(bottom_type_label_list).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    bottom_type = bottoms_list[np.argmax(probs) % len(bottoms_list)]
    score = probs[np.argmax(probs)]

    res3 = list(zip(bottom_type_label_list, probs))
    res3 = list(sorted(res3, key=lambda tup: tup[1], reverse=True))
    # print(res3)

    bottom_color_label_list = [
        "a photo of a man wearing {} {}.".format(
            color, bottoms_list[np.argmax(probs) % len(bottoms_list)]
        )
        for color in colors
    ] + [
        "a photo of a woman wearing {} {}.".format(
            color, bottoms_list[np.argmax(probs) % len(bottoms_list)]
        )
        for color in colors
    ]
    text = clip.tokenize(bottom_color_label_list).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    bottom_color = colors[np.argmax(probs) % len(colors)]
    score = probs[np.argmax(probs)]

    res4 = list(zip(bottom_color_label_list, probs))
    res4 = list(sorted(res4, key=lambda tup: tup[1], reverse=True))
    # print(res4)

    return top_type, top_color, bottom_type, bottom_color


def get_cloth_from_eshop_image(
    model, preprocess, img_file_path, tops_list, bottoms_list, colors
):
    im = cv2.imread(img_file_path)
    scale = 320 / im.shape[0]
    # cv2_imshow(cv2.resize(im, (int(im.shape[1]*scale), int(im.shape[0]*scale)), interpolation = cv2.INTER_AREA))

    image = preprocess(Image.open(img_file_path)).unsqueeze(0).to(device)

    cloth_type_label_list = ["a photo of a " + p for p in tops_list]
    cloth_type_label_list += ["a photo of " + p for p in bottoms_list]
    text = clip.tokenize(cloth_type_label_list).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    if np.argmax(probs) < len(tops_list):
        cloth_type = tops_list[np.argmax(probs)]
    else:
        cloth_type = bottoms_list[np.argmax(probs) - len(tops_list)]
    score = probs[np.argmax(probs)]

    res1 = list(zip(cloth_type_label_list, probs))
    res1 = list(sorted(res1, key=lambda tup: tup[1], reverse=True))
    # print(res1)

    if np.argmax(probs) < len(tops_list):
        cloth_color_label_list = [
            "a photo of a {} {}.".format(color, tops_list[np.argmax(probs)])
            for color in colors
        ]
    else:
        cloth_color_label_list = [
            "a photo of {} {}.".format(
                color, bottoms_list[np.argmax(probs) - len(tops_list)]
            )
            for color in colors
        ]
    text = clip.tokenize(cloth_color_label_list).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    cloth_color = colors[np.argmax(probs)]
    score = probs[np.argmax(probs)]

    res2 = list(zip(cloth_color_label_list, probs))
    res2 = list(sorted(res2, key=lambda tup: tup[1], reverse=True))
    # print(res2)

    return cloth_type, cloth_color


def get_samples(img_file_path: str):
    top_type, top_color, bottom_type, bottom_color = get_cloth_from_model_image(
        model, preprocess, img_file_path, tops_list, bottoms_list, colors
    )
    # print("{}, {}, {}, {}".format(top_type, top_color, bottom_type, bottom_color))
    return top_type, top_color, bottom_type, bottom_color

    # # find single items
    # root_dir = "/content/drive/MyDrive/fashion_sales_assistant/candidates/eshop_database"
    # samples = os.listdir(root_dir)
    # random.shuffle(samples)

    # for img_file in samples[:10]:
    #     img_file_path = os.path.join(root_dir, img_file)
    #     [cloth_type, cloth_color] = get_cloth_from_eshop_image(model, preprocess, img_file_path, tops_list, bottoms_list, colors)
    #     # print("{}, {}".format(cloth_type, cloth_color))


def get_image_paths(path):
    img_paths = []

    for f in os.listdir(path):
        tokens = f.split(".")
        extension = tokens[len(tokens) - 1]
        if os.path.isfile(os.path.join(path, f)) and (extension in IMAGE_EXTENSIONS):
            img_paths.append(os.path.join(path, f))

    img_paths.sort()

    return img_paths


def load_model_face_db(faces_dir: str, full_dir: str) -> t.Tuple[np.ndarray, dict]:
    """
    Reads faces from files and returns embedding and mapping of the order back to the filename.
    Returns:
        numpy.array of embeddings
        dict of metadata (row -> dict) to each embedding
    """

    # find all images in the folder with fashion models
    fashion_models_faces_paths = get_image_paths(FASHION_MODELS_FACES_PATH)
    fashion_models_encodings = []
    fashion_models_locations = {}
    skipped = 0
    for order, path in enumerate(tqdm(fashion_models_faces_paths)):
        face = face_recognition.load_image_file(path)
        tmp_encodings = face_recognition.face_encodings(
            face, num_jitters=10, model="large"
        )

        if len(tmp_encodings) == 0:
            skipped += 1
            continue
        else:
            fashion_models_encodings.append(tmp_encodings[0])
            with open(path.split(".")[-2] + ".json") as meta_file:
                face_meta = json.load(meta_file)
                fashion_models_locations[order - skipped] = face_meta

            face_meta["loaded_image"] = get_resized_img(
                f'{full_dir}/{face_meta["orig_filename"]}'
            )
            face_meta["current_image_path"] = f'{full_dir}/{face_meta["orig_filename"]}'

    return np.array(fashion_models_encodings), fashion_models_locations


def extract_face(path, target_path):
    MIN_FACE_AREA_SIZE = 64  # N x N pixels
    # load image
    image = face_recognition.load_image_file(path)

    # detect faces, 1x upsample is enough - we want to have detailed images, not as small as possible!
    face_locations = face_recognition.face_locations(
        image, number_of_times_to_upsample=1
    )

    # drop very small detections
    face_locations = list(
        filter(
            lambda fl: (max(fl[2] - fl[0], fl[1] - fl[3]) >= MIN_FACE_AREA_SIZE),
            face_locations,
        )
    )

    # find the largest face
    largest_size = 0
    largest_id = -1
    for fl_id, fl in enumerate(face_locations):
        pixel_count = (fl[2] - fl[0]) * (fl[1] - fl[3])
        if pixel_count > largest_size:
            largest_size = pixel_count
            largest_id = fl_id

    # if there is no face or the face is too small, skip the image
    if largest_id == -1:
        raise ValueError('skipping "{}"\n  └ no valid face detected'.format(path))

    # extract the largest face
    face_location = face_locations[largest_id]

    # add 10 % border to the face BBox
    top, right, bottom, left = face_locations[0]
    w = right - left
    h = bottom - top
    border = round(max(w, h) / 10)

    top = max(0, top - border)
    left = max(0, left - border)
    bottom = min(bottom + border, image.shape[0])
    right = min(right + border, image.shape[1])

    # cut the face out of the image
    face_crop = image[top:bottom, left:right, :]
    Image.fromarray(face_crop).save(target_path)


def crop_inmemory_imagage(image_path) -> np.ndarray:
    """
    Input: memoryview of uploaded image
    Output: face_recognition embedding of the face
    """

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp_face_file:
        extract_face(image_path, tmp_face_file.name)
        face = face_recognition.load_image_file(tmp_face_file.name)
        encodings = face_recognition.face_encodings(face, model="large")
        if len(encodings) == 0:
            raise ValueError("No encodings generated")

        return encodings[0]


# this cell define some label text
tops_list = [
    "sweater",
    "shirt",
    "suit jacket",
    "hawaiian shirt",
    "singlet",
    "cardigan",
    "jacket",
    "sleeveless shirt",
    "vest",
    "long-sleeve top",
    "polo shirt",
    "trench coat",
    "blazer",
    "T-shirt",
    "pullover",
    "wedding dress",
    "sheath dress",
    "dress",
    "bra",
    "tank top",
    "hoodie",
    "uniform",
    "coat",
    "swimsuit",
    "slip",
    "blouse",
    "evening gown",
    "overalls",
    "raincoat",
    "sweatshirt",
    "tracksuit",
    "tuxedo",
    "windbreaker",
    "jersey jacket",
    "denim jacket",
    "sleeveless dress",
]
bottoms_list = [
    "jeans",
    "shorts",
    "cargo pants",
    "dress pants",
    "skirt",
    "thong",
    "leggings",
    "pantyhose",
    "slacks",
    "sweatpants",
    "trousers",
]
colors = [
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "white",
    "black",
    "navy blue",
    "purple",
    "brown",
    "gray",
    "pink",
    "white",
    "lavender blue",
    "beige",
    "silver",
    "charcoal",
    "turquoise",
]
colors += ["light " + c for c in colors] + ["dark " + c for c in colors]

device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.isfile(f'model.{device}.pickle'):
    model, preprocess = clip.load("ViT-B/32", device=device)

    encodings, locations = load_model_face_db(
        FASHION_MODELS_FACES_PATH, FASHION_MODELS_PATH
    )

    eshop_database = {}

    for img_file in tqdm(os.listdir(ESHOP_PATH)):
        img_file_path = os.path.join(ESHOP_PATH, img_file)
        [cloth_type, cloth_color] = get_cloth_from_eshop_image(
            model, preprocess, img_file_path, tops_list, bottoms_list, colors
        )
        eshop_database[(cloth_type, cloth_color)] = img_file_path

    with open(f'model.{device}.pickle', 'wb') as f:
        pickle.dump((
            model, preprocess, encodings, locations, eshop_database
        ), f)

else:
    with open(f'model.{device}.pickle', 'rb') as f:
        model, preprocess, encodings, locations, eshop_database = pickle.load(f)


def interactive_handler(image_path):
    if not image_path:
        return

    selfie_enc = crop_inmemory_imagage(image_path)

    # find best face
    distances = np.linalg.norm(encodings - selfie_enc, axis=1)
    best_index = [i[0] for i in nsmallest(3, enumerate(distances), key=itemgetter(1))]

    similar_person_images = [locations[idx]["current_image_path"] for idx in best_index]

    recomm_clothes = [
        spl
        for idx in best_index
        for spl in get_samples(locations[idx]["current_image_path"])
    ]

    rcc = iter(recomm_clothes)
    recomm_clothes_pairs = list(zip(rcc, rcc))

    recomm_image_paths = [
        img_path
        for img_path in (eshop_database.get(r_pair) for r_pair in recomm_clothes_pairs)
        if img_path is not None
    ]

    return similar_person_images, recomm_clothes_pairs, recomm_image_paths
