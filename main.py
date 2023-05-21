from bin.dataIris import runDataIris
from bin.model import runMainModel
from bin.mlpSkelearn import runMLPSkelearn

quit = False

while (not quit):
  print(''' 
    Silakan pilih diantara opsi berikut ini untuk dijalankan
      1. Training model
      2. Training data iris
      3. Training data iris dengan MLP Skelearn
    Contoh: 3
        ''')
  while (True):
    inputUser = input(">>> ")
    if (inputUser == '1' or inputUser == '2' or inputUser == '3'):
      break
    else:
      print("input anda salah! Lakukan input kembali")
    
  if (inputUser == '1'):
    runMainModel()
  elif (inputUser == '2'):
    runDataIris()
  else :
    runMLPSkelearn()
    
  print("Apakah anda ingin keluar? (Y/N)")
  while (True):
    inputUser = input(">>> ")
    if (inputUser.upper() == 'Y'):
      print("Keluar dari program...")
      quit = True
      break
    elif (inputUser.upper() == 'N'):
      break
    else:
      print("input anda salah! Lakukan input kembali")