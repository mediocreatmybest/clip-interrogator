import argparse
import csv
import os
import requests
import torch
from PIL import Image
from clip_interrogator import Interrogator, Config, list_clip_models

def inference(interrogator, image, mode):
    image = image.convert('RGB')
    if mode == 'best':
        return interrogator.interrogate(image)
    elif mode == 'classic':
        return interrogator.interrogate_classic(image)
    else:
        return interrogator.interrogate_fast(image)

def process_single_image(interrogator, image_path, mode):
    if str(image_path).startswith('http://') or str(image_path).startswith('https://'):
        image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    if not image:
        print(f'Error opening image {image_path}')
        exit(1)
    return inference(interrogator, image, mode)

def process_folder(interrogator, folder_path, output_type, write_mode, mode, extensions=['.jpg', '.jpeg', '.png', '.webp']):
    if not os.path.exists(folder_path):
        print(f'The folder {folder_path} does not exist!')
        exit(1)

    prompts = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            _, file_ext = os.path.splitext(file)
            if file_ext.lower() in extensions:
                image_path = os.path.join(root, file)
                image = Image.open(image_path).convert('RGB')
                prompt = inference(interrogator, image, mode)
                prompts.append(prompt)
                print(prompt)

                if output_type == 'captions':
                    file_name = os.path.splitext(file)[0] + '.txt'
                    file_path = os.path.join(folder_path, file_name)

                    if write_mode == 'write':
                        write_to_file(file_path, prompt, mode='w')
                    elif write_mode == 'append':
                        write_to_file(file_path, prompt, mode='a')
                    elif write_mode == 'prepend':
                        prepend_to_file(file_path, prompt)

    if output_type == 'csv':
        csv_path = os.path.join(folder_path, 'desc.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['image', 'prompt'])
            for root, _, files in os.walk(folder_path):
                for file, prompt in zip(files, prompts):
                    _, file_ext = os.path.splitext(file)
                    if file_ext.lower() in extensions:
                        writer.writerow([file, prompt])

        print(f"\n\n\nGenerated {len(prompts)} prompts and saved to {csv_path}, enjoy!")

def write_to_file(file_path, data, mode='w', encoding='utf-8'):
    with open(file_path, mode, encoding=encoding) as f:
        if mode == 'a' and os.path.exists(file_path):
            f.write(', ')
        f.write(data)


    print(f'File saved to {file_path} in "{mode}" mode')

def prepend_to_file(file_path, data, mode='r+', encoding='utf-8'):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(data)
            return
    with open(file_path, mode, encoding=encoding) as f:
        existing_data = f.read()
        f.seek(0)
        f.write(data + ', ' + existing_data)
    print(f'File saved to {file_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clip', default='ViT-L-14/openai', metavar='ViT-H-14/laion2b_s32b_b79k',
                        help='name of CLIP model to use, ViT-L-14/openai (SD1) or ViT-H-14/laion2b_s32b_b79k (SD2)')
    parser.add_argument('-d', '--device', default='auto',
                        help='device to use (auto, cuda or cpu)')
    parser.add_argument('-f', '--folder',
                        help='path to folder of images')
    parser.add_argument('-i', '--image',
                        help='image file or url')
    parser.add_argument('-m', '--mode', default='best',
                        help='best, classic, or fast')
    parser.add_argument('-o', '--output-type', default='captions',
                        help='captions, csv', choices=['captions', 'csv'])
    parser.add_argument('-wm', '--write-mode', default='write',
                        help='file write mode (only for caption type), append files or write',
                        choices=['write', 'append', 'prepend'])

    # Additional settings based on config function
    parser.add_argument("-cml", "--caption_max_length", type=int, default=32,
                        help="Maximum length of output")
    parser.add_argument("-cm", "--caption_model_name", default='blip-large', choices=['blip-large', 'blip-base', 'blip2-2.7b', 'blip2-flan-t5-xl', 'git-large-coco'],
                        help="Name of model")
    parser.add_argument("-co", "--caption_offload", type=bool, default=True,
                        help="Offload model to CPU")
    # Additional Interrogator settings based on config function
    parser.add_argument("-flc", "--flavor_intermediate_count", type=int, default=2048,
                        help="Intermediate count for flavors, lowest value seems to be 10")
    parser.add_argument("-cs", "--chunk_size", type=int, default=2048,
                        help="CLIP chunk_size, smaller uses less vram")
    parser.add_argument("-q", "--quiet", type=bool, default=False,
                        help="Disables progress bars")
    # Additional Options to disable flavors, artists, trendings, movements, mediums
    # Flavs broken.
    #parser.add_argument("-df", "--disable-flavs", action="store_false", help="Disables flavors within captions")
    parser.add_argument("-da", "--disable-artists", action="store_false",
                        help="Disables artists within captions")
    parser.add_argument("-dm", "--disable-mediums", action="store_false",
                        help="Disables mediums within captions")
    parser.add_argument("-dmov", "--disable-movements", action="store_false",
                        help="Disables movements within captions")
    parser.add_argument("-dt", "--disable-trends", action="store_false",
                        help="Disables trendings within captions")
    parser.add_argument("--lowvram", action='store_true', help="Optimize settings for low VRAM")

    args = parser.parse_args()

    if not args.folder and not args.image:
        parser.print_help()
        exit(1)

    if args.folder is not None and args.image is not None:
        print("Specify a folder or batch processing or a single image, not both")
        exit(1)

    # validate clip model name
    models = list_clip_models()
    if args.clip not in models:
        print(f"Could not find CLIP model {args.clip}!")
        print(f"Available models: {models}")
        exit(1)

    # select device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("CUDA is not available, using CPU. Warning: this will be very slow!")
    else:
        device = torch.device(args.device)

    # generate a nice prompt
    config = Config(
        device=device,
        clip_model_name=args.clip,
        chunk_size=args.chunk_size,
        caption_max_length=args.caption_max_length,
        caption_model_name=args.caption_model_name,
        caption_offload=args.caption_offload,
        flavor_intermediate_count=args.flavor_intermediate_count,
        quiet=args.quiet,
        load_artists=args.disable_artists,
        #load_flavors=args.disable_flavs,
        load_mediums=args.disable_mediums,
        load_movements=args.disable_movements,
        load_trendings=args.disable_trends
    )
    if args.lowvram:
        config.apply_low_vram_defaults()
    interrogator = Interrogator(config)

    # process single image
    if args.image is not None:
        prompt = process_single_image(interrogator, args.image, args.mode)
        print(prompt)

    # process folder of images
    elif args.folder is not None:
        process_folder(interrogator, args.folder, args.output_type, args.write_mode, args.mode)

if __name__ == "__main__":
    main()