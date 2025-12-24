import copy
import time

import pandas as pd
from tqdm import tqdm

from SHoTClean_P1 import SHoTClean_P1
from baselines.SCREEN import SCREEN
from baselines.SpeedAcc import SpeedAcc
from tools.utils import Assist_Single
from baselines.MTCSC import MTCSC


def main():
    assist = Assist_Single()
    input_file_name = "CA/CA.csv"
    s_max = 3000
    s_min = -3000
    drate = 0.20
    seed = 1

    rmse_results = []

    for T in tqdm(range(3, 101)):
        raw_series = assist.read_data(input_file_name, ",")
        dirty_base = assist.add_noise(raw_series, drate, seed)

        dirty = copy.deepcopy(dirty_base)
        mtcsc = MTCSC(dirty, s_max, s_min, T)
        res_mtcsc = mtcsc.mainScreen()
        rms_mtcsc = assist.calc_rms(res_mtcsc)

        dirty = copy.deepcopy(dirty_base)
        screen = SCREEN(dirty, s_max, s_min, T)
        res_screen = screen.main_screen()
        rms_screen = assist.calc_rms(res_screen)

        dirty = copy.deepcopy(dirty_base)
        speedacc = SpeedAcc(dirty, T, s_max, s_min, 4000, -4000)
        res_speedacc = speedacc.main_sli_up()
        rms_speedacc = assist.calc_rms(res_speedacc)

        dirty = copy.deepcopy(dirty_base)
        shotclean = SHoTClean_P1(dirty, s_max, s_min, T, is_soft=True)
        res_shot = shotclean.mainScreen()
        rms_shotclean = assist.calc_rms(res_shot)

        rmse_results.append([
            T,
            rms_screen,
            rms_speedacc,
            rms_mtcsc,
            rms_shotclean
        ])

    df = pd.DataFrame(
        rmse_results,
        columns=['W', 'SCREEN', 'SpeedAcc', 'MTCSC', 'SHoTClean']
    )
    df.to_csv('../results/Window_Size/window_size.csv', index=False)
    print("Saved RMSE results to window_size.csv")

if __name__ == "__main__":
    main()