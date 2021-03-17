from typing import NamedTuple


class LayoutProcessingSettings(NamedTuple):
    marginalia_postprocessing: bool = False
    source_scale: bool = True  # use same scale as prediction
    # rescale_area: int = 0 # If this is given, rescale to given area
    lines_only: bool = False
    schnip_schnip: bool = False
    schnip_schnip_height_diff_factor: int = -2
    fix_line_endings: bool = True
    debug_show_fix_line_endings: bool = False

    @staticmethod
    def from_cmdline_args(args):
        return LayoutProcessingSettings(marginalia_postprocessing=args.marginalia_postprocessing,
                                        source_scale=True, lines_only=not args.layout_prediction,
                                        schnip_schnip=args.schnipschnip,
                                        debug_show_fix_line_endings=args.show_fix_line_endings and args.debug,
                                        schnip_schnip_height_diff_factor=args.height_diff_factor)