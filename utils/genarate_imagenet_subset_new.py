
labels = ['n02106382', 'n02107142', 'n02109047', 'n02110063', 'n02111889', 'n02114855', 'n02120505', 'n02123597',
          'n02125311', 'n02168699', 'n02172182', 'n02177972', 'n02206856', 'n02219486', 'n02437312', 'n02445715',
          'n02486261', 'n02492660', 'n02493509', 'n03065424', 'n03124043', 'n03124170', 'n03127747', 'n03131574',
          'n03160309', 'n03180011', 'n03188531', 'n03196217', 'n03207743', 'n03249569', 'n03290653', 'n03344393',
          'n03347037', 'n03372029', 'n03379051', 'n03388549', 'n03404251', 'n03417042', 'n03425413', 'n03445777',
          'n03447447', 'n03447721', 'n03476991', 'n03482405', 'n03483316', 'n03492542', 'n03787032', 'n03794056',
          'n03796401', 'n03825788', 'n03868863', 'n03873416', 'n03938244', 'n03944341', 'n03950228', 'n03967562',
          'n04019541', 'n04065272', 'n04136333', 'n04146614', 'n04152593', 'n04162706', 'n04209133', 'n04239074',
          'n04259630', 'n04263257', 'n04330267', 'n04355933', 'n04370456', 'n04376876', 'n04398044', 'n04418357',
          'n04429376', 'n04447861', 'n04493381', 'n04517823', 'n04532106', 'n04562935', 'n04579432', 'n07565083',
          'n07583066', 'n07584110', 'n07590611', 'n07684084', 'n07714990', 'n07717410', 'n07717556', 'n07718472',
          'n07718747', 'n07760859', 'n07831146', 'n07873807', 'n09332890', 'n09421951', 'n10148035', 'n11939491',
          'n12267677', 'n13037406', 'n13052670', 'n13054560']
import os

source_path = "/share/wenzhuoliu/torch_ds/imagenet"
target_path = "/share/wenzhuoliu/torch_ds/imagenet-subset"
# imagenet100_classes = list(imagenet_subset.keys())

for class_i in labels:
    os.symlink(os.path.join(source_path, "train", class_i), os.path.join(target_path, "train", class_i))
    os.symlink(os.path.join(source_path, "val", class_i), os.path.join(target_path, "val", class_i))
