from multiprocessing import Pool
import time

from air_traffic_model_combined import AirTrafficModelCombined


DATA_DIR = "dats"
TFR_FILE = "TFR_Lat_Lon.xlsx"


def run_mode(mode):
    start = time.time()

    model = AirTrafficModelCombined(
        data_dir=DATA_DIR,
        tfr_path=TFR_FILE,
        mode=mode,
    )

    model.run(model.max_steps)

    runtime = time.time() - start

    return {
        "mode": mode,
        "runtime": runtime,
        "conflicts": len(model.detector.get_event_log()),
    }


if __name__ == "__main__":

    modes = [
        "baseline",
        "mdp_vi",
        "pomdp",
    ]

    with Pool(processes=len(modes)) as pool:
        results = pool.map(run_mode, modes)

    print("\nRESULTS\n")

    for r in results:
        print(
            f"{r['mode']:10s}"
            f" Runtime={r['runtime']:.1f}s"
            f" Conflicts={r['conflicts']}"
        )