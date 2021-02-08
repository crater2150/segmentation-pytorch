def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, nargs="*", default=[],
                        help="Load the prediction")
    parser.add_argument("--xml_dir", type=str, help="PageXML Directory",
                        default=None)
    parser.add_arguement("--output_dir", type=str, help="Output Directory")

    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--improved_top_detection", action="store_true", help="Use improved baseline top detection")
    parser.add_argument("--marginalia_postprocessing", action="store_true", help="Enables marginalia postprocessing")

    parser.add_argument("--debug", action="store_true", help="Show a debug image")
    parser.add_argument("--show_baselines", action="store_true", help="Draws baseline to the debug image")
    parser.add_argument("--show_lines", action="store_true", help="Draws line polygons to the debug image")
    parser.add_argument("--show_layout", action="store_true", help="Draws layout regions to the debug image")
    parser.add_argument("--lines_only", action="store_true", help="Only do simple line heuristic")

    return parser.parse_args()