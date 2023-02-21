#!/usr/bin/env python3
import argparse
import csv
import open_clip
import os
import requests
import torch
from PIL import Image
from clip_interrogator import Interrogator, Config

def inference(ci, image, mode):
    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    else:
        return ci.interrogate_fast(image)

# Save function to make it a little easier to save files
def save_file(file_path, data, mode='w', encoding='utf-8', debug=False):
    """ Function to save a file, defaults to write mode """
    if not debug:
        with open(file_path, mode, encoding=encoding) as f:
            # crate seperator for append
            if mode == 'w':
                f.write(data)
            elif mode == 'a':
                data = ', ' + data
                f.write(data)
        print(f'File saved to {file_path}')
    else:
        print('Debug mode, file not saved')

# Save function to make it a little easier to prepend files
def save_file_prepend(file_path, data, mode='r+', encoding='utf-8', debug=False):
    """ Function to save with 'r+' at the start of a file seek(0) """
    if not debug:
        # Should catch missing files to prevent script stopping
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(data)
                return True
        else:
            with open(file_path, mode, encoding=encoding) as f:
                existing_data = f.read()
                f.seek(0)
                f.write(data + ', ' + existing_data)
            print(f'File saved to {file_path}')
            return True
    else:
        print('Debug mode, file not saved')
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clip', default='ViT-H-14/laion2b_s32b_b79k', metavar='ViT-H-14/laion2b_s32b_b79k',
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
                        help='file write mode (only for caption type), append files or write', choices=['write', 'append', 'prepend'])

    # Additional BLIP settings based on config function
    parser.add_argument("-bis", "--blip_image_eval_size", type=int, default=384,
                        help="Size of image for evaluation")
    parser.add_argument("-bml", "--blip_max_length", type=int, default=32,
                        help="Maximum length of BLIP model output")
    parser.add_argument("-bmt", "--blip_model_type", default='large', choices=['base','large'],
                        help="Type of BLIP model ('base' or 'large')")
    parser.add_argument("-bb", "--blip_num_beams", type=int, default=32,
                        help="Number of beams for BLIP model")
    parser.add_argument("-bo", "--blip_offload", type=bool, default=True,
                        help="Offload BLIP model to CPU")
    # Additonal Interrogator settings based on config function
    parser.add_argument("-flc", "--flavor_intermediate_count", type=int, default=2048,
                        help="Intermediate count for flavors, lowest value seems to be 10")
    parser.add_argument("-cs", "--chunk_size", type=int, default=512,
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

    # Set write mode for save function
    if args.write_mode == 'write':
        wmode = 'w'
    elif args.write_mode == 'append':
        wmode = 'a'

    if not args.folder and not args.image:
        parser.print_help()
        exit(1)

    if args.folder is not None and args.image is not None:
        print("Specify a folder or batch processing or a single image, not both")
        exit(1)

    # validate clip model name
    models = ['/'.join(x) for x in open_clip.list_pretrained()]
    if args.clip not in models:
        print(f"Could not find CLIP model {args.clip}!")
        print(f"    available models: {models}")
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
        blip_image_eval_size=args.blip_image_eval_size,
        blip_max_length=args.blip_max_length,
        blip_model_type=args.blip_model_type,
        blip_num_beams=args.blip_num_beams,
        blip_offload=args.blip_offload,
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
    ci = Interrogator(config)

    # process single image
    if args.image is not None:
        image_path = args.image
        if str(image_path).startswith('http://') or str(image_path).startswith('https://'):
            image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        if not image:
            print(f'Error opening image {image_path}')
            exit(1)
        print(inference(ci, image, args.mode))

    # process folder of images
    elif args.folder is not None:
        if not os.path.exists(args.folder):
            print(f'The folder {args.folder} does not exist!')
            exit(1)

        files = [f for f in os.listdir(args.folder) if f.endswith('.jpg') or  f.endswith('.png')]
        prompts = []
        for file in files:
            image = Image.open(os.path.join(args.folder, file)).convert('RGB')
            prompt = inference(ci, image, args.mode)
            prompts.append(prompt)
            print(prompt)

        if args.output_type == 'csv':
            csv_path = os.path.join(args.folder, 'desc.csv')
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                w.writerow(['image', 'prompt'])
                for file, prompt in zip(files, prompts):
                    w.writerow([file, prompt])

            print(f"\n\n\nGenerated {len(prompts)} and saved to {csv_path}, enjoy!")


        elif args.output_type == 'captions':
            for file, prompt in zip(files, prompts):
                file_name = os.path.splitext(file)[0] + '.txt'
                file_path = os.path.join(args.folder, file_name)

                if args.write_mode == 'write' or args.write_mode == 'append':
                    save_file(file_path=file_path, data=prompt, mode=wmode)
                elif args.write_mode == 'prepend':
                    save_file_prepend(file_path=file_path, data=prompt)


                #with open(file_path, 'w', encoding='utf-8') as f:
                #    f.write(prompt)

            print(f"\n\n\nGenerated {len(prompts)} prompts and saved to {args.folder}, enjoy!")

if __name__ == "__main__":
    main()
