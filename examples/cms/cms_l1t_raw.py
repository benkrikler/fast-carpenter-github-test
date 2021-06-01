import uproot

input_file = 'data/CMS_L1T_study.root'

f = uproot.open(input_file)
tree1 = f['l1CaloTowerEmuTree/L1CaloTowerTree']
tree2 = f['l1CaloTowerTree/L1CaloTowerTree']

caloTower = tree1['L1CaloTower']
caloTowerEmu = tree2['L1CaloTower']

for name in caloTower.keys():
    var = caloTower[name].array()
    varEmu = caloTowerEmu[name].array()
    print(name, '=', var[var!=0])
    print(name, '(emu)', '=', varEmu[varEmu != 0])

