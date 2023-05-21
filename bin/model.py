from function.generateModel import generate_model

def runMainModel():
  while (True):
    model = str(input("Masukkan model yang ingin digunakan (Silakan cek nama file yang ada pada folder model): "))
    filePath = f"model/{model}.json"
    mbgd = generate_model(filePath)

    if (mbgd == False):
      print("File tidak ditemukan pada folder model! Pastikan file yang diinput sudah berada pada folder tersebut!")
    else :
      break

  #Informasi Awal
  mbgd.visualize()

  mbgd.train()

  #Informasi Akhir
  mbgd.visualize()