from Cluster_Rater.clustering.vat import *
from skimage import filters
from PIL import Image
from skimage.morphology import square
from skimage.filters import rank
from Cluster_Rater.helpers import *
from Cluster_Rater.clustering.data_config import *
from Cluster_Rater.clustering.over_time_clustering import OverTimeClustering
import numpy as np

DATASET = generated_five

output_path = '/home/tatusch/Dokumente/KI-Projekt/finfraud/Cluster_Rater/clustering/vat_images/generated_five/'


def save_image_from_array(image_array, file_name):
    rescaled = (255.0 / image_array.max() * (image_array - image_array.min())).astype(np.uint8)
    image_final = Image.fromarray(rescaled)
    image_final.save(file_name+'.png')


def run_experiment():
    print('Running Vat Test on DATASET: ' + DATASET['Name'])
    data = load_data(DATASET)

    columns = ['ObjectID', 'Time'] + DATASET['feature_renames']
    data = data[columns]

    timestamps = data['Time'].unique()
    timestamps.sort()

    otc = OverTimeClustering(data)

    dist_matrix = otc.get_distance_matrix()
    sum_matrix = np.sum(dist_matrix, axis=0)
    # total_preferences = otc.get_total_preferences()
    # relative_preferences = otc.calc_relative_preferences(total_preferences)
    # temporal_preferences = otc.calc_temporal_preferences(relative_preferences)
    temporal_preferences = otc.get_temporal_preferences_sw(DATASET['sw'])

    for i in range(len(timestamps)):
        # dist_vat_image = vat(dist_matrix[i].copy())
        # save_image_from_array(dist_vat_image, output_path + 'distance_' + str(timestamps[i]))
        # tot_vat_image = vat(total_preferences[i].copy())
        # save_image_from_array(tot_vat_image, output_path + 'total_' + str(timestamps[i]))
        # rel_vat_image = vat(relative_preferences[i].copy())
        # save_image_from_array(rel_vat_image, output_path + 'relative_' + str(timestamps[i]))
        temp_vat_image = vat(temporal_preferences[i].copy())
        save_image_from_array(temp_vat_image, output_path + 'temporal_' + str(timestamps[i]))

        # temp_dist_matrix = (sum_matrix - dist_matrix[i]) / (len(sum_matrix) - 1)
        # temp_dist_matrix = dist_matrix[i] * temp_dist_matrix
        # temp_dist_vat_image = vat(temp_dist_matrix.copy())
        # save_image_from_array(temp_dist_vat_image, output_path + 'temporal_distance_' + str(timestamps[i]))


        # binarize vat image
        # dist_otsu_val = filters.threshold_otsu(dist_vat_image)
        # dist_vat_bin = np.where(dist_vat_image <= dist_otsu_val, 0, 255)
        # save_image_from_array(dist_vat_bin, output_path + 'distance_' + str(timestamps[i]) + '_bin')
        #
        # tot_otsu_val = filters.threshold_otsu(tot_vat_image)
        # tot_vat_bin = np.where(tot_vat_image <= tot_otsu_val, 0, 255)
        # save_image_from_array(tot_vat_bin, output_path + 'total_' + str(timestamps[i]) + '_bin')
        #
        # rel_otsu_val = filters.threshold_otsu(rel_vat_image)
        # rel_vat_bin = np.where(rel_vat_image <= rel_otsu_val, 0, 255)
        # save_image_from_array(rel_vat_bin, output_path + 'relative_' + str(timestamps[i]) + '_bin')

        temp_otsu_val = filters.threshold_otsu(temp_vat_image)
        temp_vat_bin = np.where(temp_vat_image <= temp_otsu_val, 0, 255)
        save_image_from_array(temp_vat_bin, output_path + 'temporal_' + str(timestamps[i]) + '_bin')

        # temp_dist_otsu_val = filters.threshold_otsu(temp_dist_vat_image)
        # temp_dist_vat_bin = np.where(temp_dist_vat_image <= temp_dist_otsu_val, 0, 255)
        # save_image_from_array(temp_dist_vat_bin, output_path + 'temporal_distance_' + str(timestamps[i]) + '_bin')

        # box filter with filter size n*0.04 on binarized vat image
        n = temporal_preferences[i].shape[0]
        selem = square(int(round(n*0.04)))

        # dist_filtered_vat_bin = rank.mean(dist_vat_bin, selem=selem)
        # save_image_from_array(dist_filtered_vat_bin, output_path + 'distance_' + str(timestamps[i]) + '_bin_filtered')
        #
        # tot_filtered_vat_bin = rank.mean(tot_vat_bin, selem=selem)
        # save_image_from_array(tot_filtered_vat_bin, output_path + 'total_' + str(timestamps[i]) + '_bin_filtered')
        #
        # rel_filtered_vat_bin = rank.mean(rel_vat_bin, selem=selem)
        # save_image_from_array(rel_filtered_vat_bin, output_path + 'relative_' + str(timestamps[i]) + '_bin_filtered')

        temp_filtered_vat_bin = rank.mean(temp_vat_bin, selem=selem)
        save_image_from_array(temp_filtered_vat_bin, output_path + 'temporal_' + str(timestamps[i]) + '_bin_filtered')

        # temp_dist_filtered_vat_bin = rank.mean(temp_dist_vat_bin, selem=selem)
        # save_image_from_array(temp_dist_filtered_vat_bin, output_path + 'temporal_distance_' + str(timestamps[i]) + '_bin_filtered')


run_experiment()






