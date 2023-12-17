from mercari_scraping import mercari_scraping
from zozo_scraping import zozo_scraping
from trainingModel import training
from generate_mercariPkl import generate_mercariPkl
from generate_zozoPkl import generate_zozoPkl
from similarDetection import similarDetection
from categoryDetection import categoryDetection
zozo_scraping()
generate_zozoPkl()
training()
response = categoryDetection()
mercari_scraping(response)
generate_mercariPkl()
similarDetection()



