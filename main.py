import os
import argparse
from omegaconf import OmegaConf
from src.audio import(
    device_info,
    get_stream_in,
    get_stream_out,
)
from src.streamer import Streamer

def get_parser():
    parser = argparse.ArgumentParser(
        "", # TODO
        description="Performs live speech enhancement, reading audio from "
                    "the default mic (or interface specified by --in) and "
                    "writing the enhanced version to 'Soundflower (2ch)' "
                    "(or the interface specified by --out)." # TODO
        )

    parser.add_argument(
        "-conf", "--conf", dest="conf", default="./src/conf/config.yaml",
        help="" # TODO
    )    

    parser.add_argument(
        "-itype", "--intype", dest="in_type", default="device",
        help="" # TODO 
    )
    parser.add_argument(
        "-otype", "--outtype", dest="out_type", default="device",
        help="" # TODO
    )
    parser.add_argument(
        "-i", "--in", dest="in_device", default="MacBook Pro Microphone",
        help="name or index of input interface."
    )
    parser.add_argument(
        "-o", "--out", dest="out_device", default="Soundflower (2ch)",
        help="name or index of output interface."
    )
    parser.add_argument(
        "-ifile", "--infile", dest="in_file",
        help=""  # TODO
    )
    parser.add_argument(
        "-ofile", "--outfile", dest="out_file", default="",
        help=""  # TODO
    )
    parser.add_argument(
        "-ich", "--inchannel", dest="in_channel",
        help="" # TODO
    )  
    parser.add_argument(
        "-och", "--outchannel", dest="out_channel", default="",
        help="" # TODO
    ) 
    parser.add_argument(
        "-sr", "--samplerate", dest="sample_rate", default=44100,
        help="" # TODO
    ) 
    parser.add_argument(
        "--no_compressor", action="store_false", dest="compressor",
        help="Deactivate compressor on output, might lead to clipping.")
    parser.add_argument(
        "-t", "--num_threads", type=int,
        help="Number of threads. If you have DDR3 RAM, setting -t 1 can "
             "improve performance.")
    parser.add_argument(
        "-f", "--num_frames", type=int, default=1,
        help="Number of frames to process at once. Larger values increase "
             "the overall lag, but will improve speed.")

    return parser

def main():
    print("\t Hello Streamer World")
    print("-"*30)
    
    args = get_parser().parse_args()

    if args.conf and os.path.exists(args.conf): 
        args = OmegaConf.load(args.conf)
    
    print("\t Loaded Configuration")
    print("-"*30)

    print(args)
    print("-"*30)
    
    if args.setting.in_device or args.setting.out_device:
        print("\t Currnet Device list")
        print(device_info())
        print("-"*30)

    import src.audio as audio
    print(audio.available_subtypes())
    
    streamer = Streamer(args)
    streamer.loop()

if __name__=="__main__":
    main()