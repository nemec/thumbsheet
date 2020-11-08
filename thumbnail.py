#!/usr/bin/env python3

from PIL import Image, ImageDraw, ImageFont
from datetime import timedelta
from glob import glob
import itertools
import argparse
import logging
import pathlib
import ffmpeg
import enum
import sys
import os

# ffmpeg -ss 00:00:10 -i uploads/high.webm -frames 1 -vf "select=not(mod(n\,500)),scale=480:360,tile=10x5" out.webp


logging.basicConfig(level=logging.INFO)

MARGIN = 10
HEADER_HEIGHT = 70
BACKGROUND_COLOR = (238, 238, 238) #'eeeeee'
HEADER_TEXT_COLOR = (0, 0, 0)
HEADER_TEXT_V_PADDING = 3
BORDER_WIDTH = 1
BORDER_COLOR = (0, 0, 0)
SHADOW_MARGIN = 4
SHADOW_COLOR = (0, 0, 0) #(157, 157, 157)
TIMESTAMP_TEXT_COLOR = (255, 255, 255)

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
    except KeyError:
        if count_frames:
            raise
        else:
            # Try the slower frame counting method
            logging.info("Using slower frame counting method. May take a few minutes.")
            return video_info(v, count_frames=True)

    width = int(video_stream['width'])
    height = int(video_stream['height'])
    duration = float(video_stream.get('duration') or probe['format']['duration'])

    video_codec = None
    audio_codec = None

    try:
        video_codec = (f"[{video_stream.get('codec_name','?').upper()}] "
                    f"{frac_to_fps(video_stream.get('r_frame_rate', '?'))} fps, "
                    f"{approximate_size_bits(int(video_stream.get('bit_rate', 0)))}ps")
    except ValueError:
        video_codec = 'unknown'
    if audio_stream is None:
        audio_codec = 'none'
    else:
        try:
            audio_codec = (f"[{(audio_stream.get('codec_name') or audio_stream.get('codec_tag_string','?')).upper()}] "
                        f"{audio_stream.get('sample_rate', '?')} Hz, "
                        f"{audio_stream.get('channel_layout')}, "
                        f"{approximate_size_bits(int(audio_stream.get('max_bit_rate', 0)))}")
        except ValueError:
            audio_codec = 'unknown'

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


def main(videos, x, y, width, output_dir: pathlib.Path, overwrite: bool, allow_all_extensions: bool):
    font = ImageFont.load_default()
    for f_test, size in TRY_FONTS:
        try:
            font = ImageFont.truetype(f_test, size=size)
        except OSError:
            pass
        
    videos = list(itertools.chain.from_iterable(
        [p] if p.is_file() else map(pathlib.Path, glob(str(p.expanduser()), recursive=True)) for p in videos))
    if not videos:
        logging.error("No videos matching any pattern found")
        sys.exit(1)
    
    for v in videos:
        if not v.is_file():  # skip directories
            continue
        if not allow_all_extensions and v.suffix.lower() not in ALLOWED_FILE_EXTENSIONS:
            logging.warning(f'Skipping file {v} with non-video extension. Add '
                             '--all argument to attempt to create thumbnails '
                             'for the file anyway.')
            continue
        out_file = v.with_suffix('.jpg')
        if output_dir is not None:
            out_file = output_dir / out_file.name
        if not overwrite and out_file.exists():
            result = input(f'Thumbnail file "{out_file}" exists. Overwrite? [Yn]')
            if result and not result.lower().startswith('y'):
                continue

        logging.info("Creating thumbnails for video %s", v)
        logging.debug("Reading video metadata")
        try:
            w, h, duration_s, num_frames, video_codec, audio_codec = video_info(v)
        except ValueError:
            continue
        except KeyError as e:
            logging.error('Could not find metadata for video', exc_info=e)
            continue
        logging.debug("Video dimensions: %dx%d", w, h)
        logging.info("Extracting %d frames out of %d", min(num_frames, x*y), num_frames)
        
        w_scaled = ((width - MARGIN) // x) - MARGIN
        h_scaled = int(h * w_scaled / w)
        
        fps = max(1, min(num_frames, num_frames // (x*y+2)))

        # for large videos, faster to run ffmpeg multiple times than evaluate each frame
        if num_frames > LARGE_VIDEO_FRAME_COUNT:
            logging.debug(f'Using multi-process heuristic because frame count {num_frames} > {LARGE_VIDEO_FRAME_COUNT}')
            frames = bytearray()
            skip = duration_s / (x*y+2)
            try:
                for i in range(1, x*y+1):
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

        frame_size = w_scaled * h_scaled * 3
        height = HEADER_HEIGHT + (MARGIN + h_scaled) * y
        base = Image.new('RGB', (width, height), BACKGROUND_COLOR)

        filename = v.name
        file_size = approximate_size_bytes(os.path.getsize(v))
        duration = format_time(duration_s)        

        header = ImageDraw.Draw(base)
        _, h_height = header.textsize('SAMPLE', font=font)
        header.text((MARGIN, MARGIN), f'Filename: {filename}', font=font, fill=HEADER_TEXT_COLOR)
        header.text((MARGIN, MARGIN + (h_height + HEADER_TEXT_V_PADDING)*1), f'Duration: {duration}', font=font, fill=HEADER_TEXT_COLOR)
        header.text((MARGIN, MARGIN + (h_height + HEADER_TEXT_V_PADDING)*2), f'File size: {file_size}', font=font, fill=HEADER_TEXT_COLOR)

        dimensions = f'Dimensions: {w}x{h} px'
        h_width, _ = header.textsize(dimensions, font=font)
        header.text((width-h_width-MARGIN, MARGIN), dimensions, font=font, fill=HEADER_TEXT_COLOR)
        if video_codec:
            video_codec = f'Video: {video_codec}'
            h_width, _ = header.textsize(video_codec, font=font)
            header.text((width-h_width-MARGIN, MARGIN + (h_height + HEADER_TEXT_V_PADDING) * 1), video_codec, font=font, fill=HEADER_TEXT_COLOR)
        if audio_codec:
            audio_codec = f'Audio: {audio_codec}'
            h_width, _ = header.textsize(audio_codec, font=font)
            header.text((width-h_width-MARGIN, MARGIN + (h_height + HEADER_TEXT_V_PADDING) * 2), audio_codec, font=font, fill=HEADER_TEXT_COLOR)

        
        for y_idx in range(y):
            for x_idx in range(x):
                frame_num = x_idx + y_idx * x
                if frame_num > num_frames - 2:
                    break
                frame = out[frame_num * frame_size : (frame_num+1) * frame_size]
                
                shadow = Image.new('RGB', (w_scaled, h_scaled), SHADOW_COLOR)
                base.paste(
                    shadow,
                    (MARGIN + (MARGIN + w_scaled) * x_idx + SHADOW_MARGIN,
                    HEADER_HEIGHT + (MARGIN + h_scaled) * y_idx + SHADOW_MARGIN))
                
                border = Image.new('RGB',
                    (w_scaled + 2*BORDER_WIDTH, h_scaled + 2*BORDER_WIDTH), BORDER_COLOR)
                base.paste(
                    border,
                    (MARGIN + (MARGIN + w_scaled) * x_idx - BORDER_WIDTH,
                    HEADER_HEIGHT + (MARGIN + h_scaled) * y_idx - BORDER_WIDTH))
                
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

                base.paste(
                    thumb,
                    (MARGIN + (MARGIN + w_scaled) * x_idx,
                    HEADER_HEIGHT + (MARGIN + h_scaled) * y_idx))
                
        base.save(out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--all', action='store_true',
                        help='Allow all file extensions. By default, will skip '
                             'non-video extensions which can help when globbing '
                             'folders with video in multiple formats and '
                             'non-video files that should be skipped.')
    parser.add_argument('-d', '--dimensions', type=parse_dimensions, default={'x': 5, 'y': 5},
                        help="Number of panels in WxH format. Horizontal first, then vertical")
    parser.add_argument('-w', '--width', type=int, default=1024,
                        help='Width in pixels of the output image')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help="Automatically overwrite an existing thumbnail file if it exists, do not prompt.")
    parser.add_argument('--output-dir', type=pathlib.Path,
                        help="Alternate output directory for thumbnail.")
    parser.add_argument('videos', nargs='+', type=pathlib.Path, metavar='V')
                        
    args = parser.parse_args()

    if args.output_dir and not args.output_dir.is_dir():
        logging.fatal(f'Output directory "{args.output_dir}" does not exist or is not a directory.')
        sys.exit(1)

    main(args.videos, args.dimensions['x'], args.dimensions['y'],
         args.width, args.output_dir, args.overwrite, args.all)
