from enum import Enum
from typing import NamedTuple


class LayoutProcessingMethod(Enum):
    LINES_ONLY = "linesonly"
    ANALYSE = "analyse"
    ANALYSE_SCHNIPSCHNIP = "analyse+schnipschnip"
    FULL = "full"
    FULL_REGIONSONLY = "full+regionsonly"


class LayoutProcessingSettings(NamedTuple):
    layout_method: LayoutProcessingMethod = LayoutProcessingMethod.FULL
    marginalia_postprocessing: bool = False
    source_scale: bool = True  # use same scale as prediction
    # rescale_area: int = 0 # If this is given, rescale to given area
    schnip_schnip_height_diff_factor: int = -2
    fix_line_endings: bool = True
    debug_show_fix_line_endings: bool = False

    @staticmethod
    def from_cmdline_args(args):
        if args.layout_method:
            layout_method = LayoutProcessingMethod(args.layout_method)
        elif args.layout_prediction:
            if args.schnipschnip:
                layout_method = LayoutProcessingMethod.ANALYSE_SCHNIPSCHNIP
            else:
                layout_method = LayoutProcessingMethod.ANALYSE
        else:
            layout_method = LayoutProcessingMethod.LINES_ONLY

        return LayoutProcessingSettings(marginalia_postprocessing=args.marginalia_postprocessing,
                                        source_scale=True, layout_method=layout_method,
                                        debug_show_fix_line_endings=args.show_fix_line_endings and args.debug,
                                        schnip_schnip_height_diff_factor=args.height_diff_factor)
