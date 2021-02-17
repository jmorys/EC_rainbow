import argparse
import os
from skimage import io, img_as_ubyte, img_as_float32
from python_approach.unet_tools import *
from skimage import io, img_as_ubyte, img_as_float32, segmentation, filters, morphology
from python_approach.watershed_tools import *
import tqdm
import skfmm


def main(config):
    images_cells = []
    images_vessels = []

    names = []
    if os.path.isfile(config.original_images):
        names.append(config.original_images.split("\\")[-1][:-4])
    else:
        files = [file[:-4] for file in os.listdir(config.original_images) if file.endswith(".tif")]
        assert len(files) > 0
        names.extend(list(files))

    outputfolder = config.out_location

    proper_cont = ["cell_parametres.csv", "edge_parametres.csv", "relabeled.tif"]

    if len(names) == 1 and names[0] == outputfolder.split("\\")[-1]:
        if len(set(proper_cont).difference(os.listdir(outputfolder))) == 0:
            print("results already present")
            return None
        else:
            outputfolder = "\\".join(outputfolder.split("\\")[:-1])
    elif len(set(names).intersection(os.listdir(outputfolder))) != 0:
        suspects = list(set(names).intersection(os.listdir(outputfolder)))
        suspects_rev = []
        for suspect in suspects:
            out_f = os.path.join(outputfolder, suspect)
            if len(set(proper_cont).difference(os.listdir(out_f))) == 0:
                suspects_rev.append(suspects_rev)
        names = list(set(names).difference(suspects_rev))
    if len(names) == 0:
        print("results already present")
        return None


    images_cells = [[]]*len(names)
    images_vessels = [[]]*len(names)
    if os.path.isfile(config.cells_input) and os.path.isfile(config.vessels_input):
        images_cells[0] = config.cells_input
        images_vessels[0] = config.vessels_input
    elif os.path.isdir(config.cells_input) and os.path.isdir(config.vessels_input):
        cell_f = os.listdir(config.cells_input)
        cell_f = [file.rstrip("_cells.tif") for file in cell_f if file.endswith("_cells.tif")]
        vess_f = os.listdir(config.vessels_input)
        vess_f = [file.rstrip("_vessels.tif") for file in vess_f if file.endswith("_vessels.tif")]

        for n, name in enumerate(names):
            cel = list(set(cell_f).intersection([name]))
            ves = list(set(vess_f).intersection([name]))
            if len(cel) == 0 or len(ves) == 0:
                print("missing inputs")
                return None
            else:
                images_cells[n] = os.path.join(config.cells_input, cel[0] + "_cells.tif")
                images_vessels[n] = os.path.join(config.vessels_input, ves[0] + "_vessels.tif")
    else:
        print("Stop it. Get some help.")

    sato1 = CreateSato(range(4, 17, 4))
    sato2 = CreateSato([1, 3])

    for i, info in enumerate(zip(names, images_vessels, images_cells)):
        name, vessels_n, cells_n = info
        print("Procesing image {} out of {}".format(i + 1, len(names)))
        vessels = io.imread(vessels_n, plugin="tifffile")
        cells = io.imread(cells_n, plugin="tifffile")
        img = np.concatenate((cells, vessels[:, :, np.newaxis]), axis=-1)
        del cells, vessels
        r = os.path.join(outputfolder, name)
        if not os.path.isdir(r):
            os.mkdir(r)
        relabeled, cell_params, edge_params = get_graph(img, sato1=sato1, sato2=sato2)
        np.savetxt(os.path.join(r, "cell_parametres.csv"), cell_params, delimiter=",")
        np.savetxt(os.path.join(r, "edge_parametres.csv"), edge_params, delimiter=",")
        io.imsave(os.path.join(r, "relabeled.tif"), img_as_float32(relabeled), plugin="tifffile")
        del relabeled, cell_params, edge_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cells_input', type=str)
    parser.add_argument('--original_images', type=str)
    parser.add_argument('--vessels_input', type=str)
    parser.add_argument('--out_location', type=str)
    config = parser.parse_args()
    main(config)
