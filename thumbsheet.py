#!/usr/bin/env python3

from PIL import Image, ImageDraw, ImageFont
from datetime import timedelta
from glob import glob
import itertools
import argparse
import logging
import hashlib
import pathlib
import ffmpeg
import shutil
import enum
import sys
import os


logging.basicConfig(level=logging.INFO)

MARGIN = 10
BACKGROUND_COLOR = (238, 238, 238) #'eeeeee'
HEADER_TEXT_COLOR = (0, 0, 0)
HEADER_TEXT_V_PADDING = 3
BORDER_WIDTH = 1
BORDER_COLOR = (0, 0, 0)
SHADOW_MARGIN = 4
SHADOW_COLOR = (0, 0, 0) #(157, 157, 157)
TIMESTAMP_TEXT_COLOR = (255, 255, 255)

DEFAULT_FILENAME_PATTERN = '{stem}'

OUTPUT_FILE_EXTENSION = '.png'  # '.jpg'

LARGE_VIDEO_FRAME_COUNT = 10000

TRY_FONTS = [
    ('DejaVuSans.ttf', 14),
    ('Helvetica.ttf', 14)
]

# https://en.wikipedia.org/wiki/Video_file_format#List_of_video_file_formats
ALLOWED_FILE_EXTENSIONS = set([
    '.webm', '.mkv', '.flv','.vob', '.ogv', '.ogg', '.drc', '.gif', '.gifv',
    '.mng', '.avi', '.mts', '.m2ts', '.ts', '.mov', '.qt', '.wmv', '.yuv',
    '.rm', '.rmvb', '.viv', '.asf', '.amv', '.mp4', '.m4p', '.m4v', '.mpg',
    '.mp2', '.mpeg', '.mpe', '.mpv', '.m2v', '.m4v', '.svi', '.3gp', '.3g2',
    '.mxf', '.roq', '.nsv', '.f4v', '.f4p', '.f4a', '.f4b'
])

def md5_hash_file(file: pathlib.Path):
    try:
        with open(file, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError:
        pass
    return None

def sha1_hash_file(file: pathlib.Path):
    try:
        with open(file, 'rb') as f:
            return hashlib.sha1(f.read()).hexdigest()
    except OSError:
        pass
    return None

def parse_sort_direction(val):
    print(val)
    if val is not None and val not in '+-':
        raise ValueError("Sorting direction must be +, -, or empty string")
    return val

def parse_dimensions(val):
    x, sep, y = val.partition('x')
    return {'x': abs(int(x)), 'y': abs(int(y))}
    
    
def format_time(sec):
    hours = int(sec // 3600)
    sec = sec % 3600
    minutes = int(sec // 60)
    sec = int(sec % 60)
    return f'{hours:02d}:{minutes:02d}:{sec:02d}'
    
    
def video_info(v, count_frames=False):
    if count_frames:
        probe = ffmpeg.probe(str(v), count_frames=None)
    else:
        probe = ffmpeg.probe(str(v))
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
    if video_stream is None:
        logging.error(f'No video stream in file "{v}". Skipping.')
        raise ValueError()
        
    try:
        num_frames = int(video_stream.get('nb_frames') or video_stream['nb_read_frames'])
    except (KeyError, ValueError):
        if count_frames:
            raise
        else:
            # Try the slower frame counting method
            logging.info("Using slower frame counting method. May take a few minutes.")
            return video_info(v, count_frames=True)

    width = int(video_stream['width'])
    height = int(video_stream['height'])
    duration = float(video_stream.get('duration') or probe['format']['duration'])

    video_codec = 'unknown'
    audio_codec = 'none'

    video_codec = (f"[{video_stream.get('codec_name','?').upper()}] "
                   f"{frac_to_fps(video_stream.get('r_frame_rate', '?'))} fps, "
                   f"{approximate_size_bits(int(video_stream.get('bit_rate', 0)))}ps")
    if audio_stream is not None:
        audio_codec = (f"[{(audio_stream.get('codec_name') or audio_stream.get('codec_tag_string','?')).upper()}] "
                       f"{audio_stream.get('sample_rate', '?')} Hz, "
                       f"{audio_stream.get('channel_layout')}, "
                       f"{approximate_size_bits(int(audio_stream.get('max_bit_rate', 0)))}")

    return (width, height, duration, num_frames, video_codec, audio_codec)


UNITS = {1000: ['KB', 'MB', 'GB'],
         1024: ['KiB', 'MiB', 'GiB']}

def approximate_size_bytes(size, mult=1000):
    for unit in UNITS[mult]:
        size = size / mult
        if size < mult:
            return '{0:.2f} {1}'.format(size, unit)

UNITS_BITS = {1000: ['Kb', 'Mb', 'Gb'],
              1024: ['Kib', 'Mib', 'Gib']}

def approximate_size_bits(size, mult=1000):
    for unit in UNITS_BITS[mult]:
        size = size / mult
        if size < mult:
            return '{0:.1f} {1}'.format(size, unit)


def frac_to_fps(frac: str):
    if frac is None:
        return None
    lft, sep, rt = frac.partition('/')
    if not sep:
        return frac
    try:
        return '{0:0.1f}'.format(int(lft)/int(rt))
    except ValueError:
        return frac


def main(videos, x, y, width, output_dir: pathlib.Path,
         overwrite: bool, allow_all_extensions: bool, pattern: str, sort: str, hash_type: str):
    font = ImageFont.load_default()
    for f_test, size in TRY_FONTS:
        try:
            font = ImageFont.truetype(f_test, size=size)
        except OSError:
            pass
        
    videos = list(f for f in
        itertools.chain.from_iterable(
            [p]
                if p.is_file()
                else map(pathlib.Path, glob(str(p.expanduser()), recursive=True))
            for p in videos)
        if f.is_file())
    if not videos:
        logging.error("No videos matching any pattern found")
        sys.exit(1)
    
    # We can set scaled width here, as it's defined in the CLI, but
    # must wait to probe video dimensions before scaled height can
    # be calculated.
    w_scaled = ((width - MARGIN) // x) - MARGIN

    if w_scaled <= 0:
        logging.fatal(f'Width setting {width} is too small to generate the requested number of previews.')
        sys.exit(1)
    
    if sort:
        videos = sorted(videos, key=str.casefold)
    for idx, v in enumerate(videos):
        if not allow_all_extensions and v.suffix.lower() not in ALLOWED_FILE_EXTENSIONS:
            logging.warning(f'Skipping file {v} with non-video extension. Add '
                             '--all argument to attempt to create thumbnails '
                             'for the file anyway.')
            continue

        try:
            out_file = v.parent / (pattern.format(stem=v.stem, name=v.name, idx=idx+1) + OUTPUT_FILE_EXTENSION)
        except KeyError as e:
            logging.error(f'Invalid pattern (-p). Unable to output thumb file.', exc_info=e)
            sys.exit(1)
            
        if output_dir is not None:
            out_file = output_dir / out_file.name
        if not overwrite and out_file.exists():
            result = input(f'Thumbnail file "{out_file}" exists. Overwrite? [Yn]')
            if result and not result.lower().startswith('y'):
                continue

        logging.info("Saving thumbnails for video %s to %s", v, out_file)
        logging.debug("Reading video metadata")
        try:
            w, h, duration_s, num_frames, video_codec, audio_codec = video_info(v)
        except ValueError:
            continue
        except KeyError as e:
            logging.error('Could not find metadata for video', exc_info=e)
            continue
        logging.debug("Video dimensions: %dx%d", w, h)
        
        thumb_count = min(num_frames, x*y)
        bracketed_thumb_count = min(num_frames, (x*y)+2)

        logging.debug("Extracting %d frames out of %d", thumb_count, num_frames)
        
        h_scaled = int(h * w_scaled / w)

        fps = max(1, num_frames // bracketed_thumb_count)

        # for large videos, faster to run ffmpeg multiple times than evaluate
        # each frame in one pass through the video. The -ss can skip straight
        # to a point in time, but cannot be used multiple times in one ffmpeg
        # call
        if num_frames > LARGE_VIDEO_FRAME_COUNT:
            logging.debug(f'Using multi-process method because frame count {num_frames} > {LARGE_VIDEO_FRAME_COUNT}')
            frames = bytearray()
            skip = duration_s / bracketed_thumb_count
            try:
                for i in range(1, bracketed_thumb_count):
                    out, _ = (
                        ffmpeg
                        .input(str(v), ss=skip*i, guess_layout_max=0)
                        .output('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{w_scaled}x{h_scaled}', vframes=1)
                        .global_args("-hide_banner")
                        .global_args("-loglevel", "warning")
                        .run(capture_stdout=True)
                    )
                    frames.extend(out)
            except ffmpeg._run.Error as e:
                logging.error("Error collecting frames from video", exc_info=e)
                continue
            out = bytes(frames)
        else:
            try:
                out, _ = (
                    ffmpeg
                    .input(str(v), r=fps)
                    .filter('select', 'gte(n,{})*not(mod(n-1,{}))'.format(fps, fps))
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24', r=1, s=f'{w_scaled}x{h_scaled}')
                    .global_args("-hide_banner")
                    .global_args("-loglevel", "warning")
                    .run(capture_stdout=True)
                )
            except ffmpeg._run.Error as e:
                logging.error("Error collecting frames from video", exc_info=e)
                continue
        
        # using all caps to gauge the max height of a line of text
        _, text_height = font.getsize('SAMPLE')

        max_header_text_lines = 4  # make sure to sync this with the code that comes after
        header_height = 2*MARGIN + max_header_text_lines * (text_height + HEADER_TEXT_V_PADDING)

        # * 3 to cover R, G, and B bytes
        frame_size = w_scaled * h_scaled * 3
        height = header_height + (MARGIN + h_scaled) * y
        base_layer = Image.new('RGB', (width, height), BACKGROUND_COLOR)

        duration = format_time(duration_s)        
        file_size = approximate_size_bytes(os.path.getsize(v))
        hash = None
        if hash_type == 'md5':
            hash = f'MD5 Hash: {md5_hash_file(v)}'
        elif hash_type == 'sha1':
            hash = f'SHA1 Hash: {sha1_hash_file(v)}'

        header = ImageDraw.Draw(base_layer)

        header.text((MARGIN, MARGIN), f'Filename: {v.name}', font=font, fill=HEADER_TEXT_COLOR)
        header.text((MARGIN, MARGIN + (text_height + HEADER_TEXT_V_PADDING)*1), f'Duration: {duration}', font=font, fill=HEADER_TEXT_COLOR)
        header.text((MARGIN, MARGIN + (text_height + HEADER_TEXT_V_PADDING)*2), f'File size: {file_size}', font=font, fill=HEADER_TEXT_COLOR)

        dimensions = f'Dimensions: {w}x{h} px'
        h_width, _ = header.textsize(dimensions, font=font)
        header.text((width-h_width-MARGIN, MARGIN), dimensions, font=font, fill=HEADER_TEXT_COLOR)
        if video_codec:
            video_codec = f'Video: {video_codec}'
            h_width, _ = header.textsize(video_codec, font=font)
            header.text((width-h_width-MARGIN, MARGIN + (text_height + HEADER_TEXT_V_PADDING) * 1), video_codec, font=font, fill=HEADER_TEXT_COLOR)
        if audio_codec:
            audio_codec = f'Audio: {audio_codec}'
            h_width, _ = header.textsize(audio_codec, font=font)
            header.text((width-h_width-MARGIN, MARGIN + (text_height + HEADER_TEXT_V_PADDING) * 2), audio_codec, font=font, fill=HEADER_TEXT_COLOR)
        if hash:
            h_width, _ = header.textsize(hash, font=font)
            header.text((width-h_width-MARGIN, MARGIN + (text_height + HEADER_TEXT_V_PADDING) * 3), hash, font=font, fill=HEADER_TEXT_COLOR)

        
        for y_idx in range(y):
            for x_idx in range(x):
                frame_num = x_idx + y_idx * x
                if frame_num > num_frames - 2:
                    break
                frame = out[frame_num * frame_size : (frame_num+1) * frame_size]
                
                shadow = Image.new('RGB', (w_scaled, h_scaled), SHADOW_COLOR)
                base_layer.paste(
                    shadow,
                    (MARGIN + (MARGIN + w_scaled) * x_idx + SHADOW_MARGIN,
                    header_height + (MARGIN + h_scaled) * y_idx + SHADOW_MARGIN))
                
                border = Image.new('RGB',
                    (w_scaled + 2*BORDER_WIDTH, h_scaled + 2*BORDER_WIDTH), BORDER_COLOR)
                base_layer.paste(
                    border,
                    (MARGIN + (MARGIN + w_scaled) * x_idx - BORDER_WIDTH,
                    header_height + (MARGIN + h_scaled) * y_idx - BORDER_WIDTH))
                
                thumb = Image.frombytes('RGB', (w_scaled, h_scaled), frame, 'raw')
                
                ts = duration_s * (frame_num + 1) / ((x * y) + 2)
                text = format_time(ts)
                draw = ImageDraw.Draw(thumb)
                txt_w, txt_h = draw.textsize(text, font=font)
                txt_x = w_scaled - txt_w - 5
                txt_y = h_scaled - txt_h - 5

                # jumble the text by 1px in all directions to create a
                # border keeping the text readable
                draw.text((txt_x-1, txt_y-1), text, font=font, fill=SHADOW_COLOR)
                draw.text((txt_x+1, txt_y-1), text, font=font, fill=SHADOW_COLOR)
                draw.text((txt_x-1, txt_y+1), text, font=font, fill=SHADOW_COLOR)
                draw.text((txt_x+1, txt_y+1), text, font=font, fill=SHADOW_COLOR)

                # now draw the text over it
                draw.text((txt_x, txt_y), text, font=font, fill=TIMESTAMP_TEXT_COLOR)

                base_layer.paste(
                    thumb,
                    (MARGIN + (MARGIN + w_scaled) * x_idx,
                    header_height + (MARGIN + h_scaled) * y_idx))
                
        base_layer.save(out_file)


if __name__ == '__main__':
    if not shutil.which('ffprobe'):
        logging.fatal('ffprobe application cannot be found on PATH. Is ffmpeg installed?')
        sys.exit(1)
    if not shutil.which('ffmpeg'):
        logging.fatal('ffmpeg application cannot be found on PATH. Is ffmpeg installed?')
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--all', action='store_true',
                        help='Allow all file extensions. By default, will skip '
                             'non-video extensions which can help when globbing '
                             'folders with video in multiple formats and '
                             'non-video files that should be skipped.')
    parser.add_argument('-d', '--dimensions', type=parse_dimensions, default={'x': 5, 'y': 5},
                        help="Number of panels in WxH format. Horizontal first, then vertical")
    parser.add_argument('-w', '--width', type=int, default=1024,
                        help='Width in pixels of the entire thumbsheet. Height '
                             'and width of the thumbnails is calculated based on '
                             "the dimensions of the video. It's not recommended "
                             'to set this value < 500, and it may fail entirely '
                             'if the width is not large enough to accomodate '
                             'the margin size.')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help="Automatically overwrite an existing thumbnail file if it exists, do not prompt.")
    parser.add_argument('-p', '--pattern', type=str, default=DEFAULT_FILENAME_PATTERN,
                        help='Filename pattern in Python keyword formatting syntax. '
                            f'Defaults to "{DEFAULT_FILENAME_PATTERN}". '
                             'Options: '
                             '{stem} = original filename without extension, '
                             '{name} = original filename with extension, '
                             '{idx} = index in thumbnail queue. starts at 1 (useful when thumbnailing multiple files)'
                             )
    parser.add_argument('-s', '--sort', action='store_true',
                        help='Sort input by filename before '
                             'processing. Mostly useful when combined with '
                             '{idx} pattern. By default, left unsorted.')
    parser.add_argument('--hash', choices={'none', 'md5', 'sha1'}, default='md5',
                        help='Choose the hash algorithm to use in the description. '
                             'If "none" is chosen, will not hash (this will make '
                             'thumbnail generation faster)')
    parser.add_argument('--output-dir', type=pathlib.Path,
                        help="Alternate output directory for thumbnail.")
    parser.add_argument('videos', nargs='+', type=pathlib.Path, metavar='V')
                        
    args = parser.parse_args()

    if args.output_dir and not args.output_dir.is_dir():
        logging.fatal(f'Output directory "{args.output_dir}" does not exist or is not a directory.')
        sys.exit(1)

    main(args.videos, args.dimensions['x'], args.dimensions['y'],
         args.width, args.output_dir, args.overwrite, args.all, args.pattern, args.sort, args.hash)
