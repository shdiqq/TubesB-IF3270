from function.generateModel import generate_model
from function.visualize import visualize

model = str(input("Masukkan model yang ingin digunakan (Silakan cek nama file yang ada pada folder model): "))

filePath = f"model/{model}.json"
mbgd = generate_model(filePath)

#Informasi Awal
visualize(mbgd)

mbgd.train()

#Informasi Akhir
visualize(mbgd)