	import pandas as pd
	import math
	import torchvision.models as models
	import torch
	import torch.nn.functional as F
	import torchvision
	from torch import nn
	from torch.utils.data import Dataset, DataLoader
	from sklearn.model_selection import train_test_split
	from skimage import io, transform
	from pathlib import Path
	from torchvision import transforms
	import numpy as np
	from PIL import Image, ImageFile
	from tqdm import tqdm
	
	
	#############################################################
	###################### Funciones de ayuda ###################
	#############################################################
	'''
	Separa los datos de un dataframe en test/train/valid
	'''
	def crear_conjuntos(df):
		train, test = train_test_split(df, test_size=0.2, shuffle=False)
		train, valid = train_test_split(train, test_size=0.2)
		return train, test, valid
	
	'''
	Funcion que guarda un modelo
	'''
	def guardarModelo(modelo, optimizador, nombre):
	torch.save({
		'model_state_dict': modelo.state_dict(),
		'optimizer_state_dict': optimizador.state_dict(),
	}, nombre)
	
	'''
	Crear modelo 
	'''
	def Resnet50(bloquado=False, num_clases):
		resnet = models.resnet50(progress=True, pretrained=True) # Carga de Resnet50
		if (bloquado=False):
			for param in resnet.parameters():
				param.requires_grad = True # Congelado de parametros
		resnet.fc = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(in_features=resnet.fc.in_features, out_features=num_clases),nn.Sigmoid()) # Capas ayadidas al final
		return resnet
		
	'''
	Entrenamiento del modelo
	'''
	def train(optimizador, modelo, perdida, epocas, carga_train, carga_valid, tipo):
		for epoch in range(1, epocas + 1): # Para el numero de epocas
			train_loss = 0.0
			valid_loss = 0.0
			counter1 = math.ceil(len(train_loader.dataset)/batch_size) #Numero de lotes de entrenamiento
			counter2 = math.ceil(len(valid_loader.dataset)/batch_size) #Numero de lotes de validacion
			
			modelo.train() # Ponemos el modelo en modo entrenamiento
			for i,(dato, etiqueta) in tqdm(enumerate(carga_train), total = counter1): #Recorremos todos los lotes
				dato, etiqueta = dato.cuda(), etiqueta.cuda() # Llevamos los datos y las etiquetas a la grafica
				optimizador.zero_grad() # Ponemos a 0 los gradientes previamente calculados
				output = modelo(dato) # Propagacion para adelante: Resultados de la red
				loss = perdida(output, etiqueta) #Perdida del lote
				train_loss+=loss.item() # Sumamos la perida del lote a la suma de perdidas de entrenamiento
				loss.backward() # Retro propagacion del error
				optimizador.step() #Ajuste de los pesos
			
			modelo.eval() # Ponemos el modelo en modo evaluacion (Desactiva dropout)
			with torch.no_grad(): # Congelamos los pesos y eliminamos el dropout
				for i,(dato, etiqueta) in tqdm(enumerate(carga_valid), total = counter2):
					dato, etiqueta = dato.cuda(), etiqueta.cuda() # Llevamos los datos y las etiquetas a la grafica
					output = modelo(dato) # Propagacion para adelante: Resultados de la red
					loss = perdida(output, etiqueta) #Perdida del lote
					valid_loss += loss.item() # Sumamos la perida del lote a la suma de perdidas de validacion
					
	'''
	Test del modelo
	'''
	def test(modelo, carga_test):
	    soterrado_predict, grafiti_predict, quemado_predict = [], [], []
		modelo.eval() 
		with torch.no_grad():
			for images, labels in carga_test:
				images, labels = images.to(device), labels.to(device)  # Llevamos los datos y las etiquetas a la grafica
				outputs = modelo(images) # Propagacion para adelante: Resultados de la red
				
				# Resultados finales de Soterrados => se guarda en soterrado_predict
				if int(outputs[:,0]>0.9): soterrado_predict.append(1)
				elif int(outputs[:,0]<0.1): soterrado_predict.append(0)
				else: soterrado_predict.append(int(not labels[:, 0]))
				
				# Resultados finales de Grafiti => se guarda en grafiti_predict
				if int(outputs[:,1]>0.9): grafiti_predict.append(1)
				elif int(outputs[:,1]<0.1): grafiti_predict.append(0)
				else: grafiti_predict.append(int( not labels[:,1]))
				
				# Resultados finales de Quemado => se guarda en quemado_predict
				if int(outputs[:,2]>0.9): quemado_predict.append(1)
				elif int(outputs[:,2]<0.1): quemado_predict.append(0)
				else: quemado_predict.append(int( not labels[:,1]))
				
		return soterrado_predict, grafiti_predict, quemado_predict

	#############################################################
	######################## Clases  ############################
	#############################################################
	
	'''
	Crea el cargador de las imagenes y etiquetas
	'''
	class ContenedorDataset(Dataset):
		#Se guarda como atributo el conjunto de datos y transformaciones
		def __init__(self, data, transform):
			super().__init__()
			self.data = data  
			self.transform = transform  
		
		def __len__(self):
			return len(self.data)
	
		def __getitem__(self,index):
			# Carga y apertura de laa imagen
			img_path = self.data[index, 0]
			image = Image.open(img_path)
			# Hacer las transformaciones necesarias y crear la etiqueta
			return self.transform(image), torch.tensor([self.data[index, 1], self.data[index, 2], self.data[index, 3]],dtype=torch.float32)
	
	'''
	Crea La perdida Margin Loss
	'''
	class MarginLoss(nn.Module):
		def __init__(self, size_average=False, loss_lambda=0.5):
			super(MarginLoss, self).__init__()
			self.size_average = size_average
			self.m_plus = 0.9
			self.m_minus = 0.1
			self.loss_lambda = loss_lambda
		
		def forward(self, inputs, labels):
			L_k = labels * F.relu(self.m_plus - inputs)**2 + self.loss_lambda * (1 - labels) * F.relu(inputs - self.m_minus)**2
			L_k = L_k.sum(dim=1)
			if self.size_average:
				return L_k.mean()
			else:
				return L_k.sum()
				
				
	##############################################################
	################### PROGRAMA PRINCIPAL #######################
	##############################################################
	def main():
		#Filtrado y preparacion de los datos
		df = pd.read_csv(src, usecols= ['FOTO','SOTERRADO', 'GRAFITI', 'QUEMADO'], sep=';')
		df['SOTERRADO'].map(lambda x : 1 if x=='Si' else 0)
		df['GRAFITI'].map(lambda x : 1 if x=='Si' else 0)
		df['QUEMADO'].map(lambda x : 1 if x=='Si' else 0)
		df_si = df[df['SOTERRADO'].str.match(1) | df['GRAFITI'].str.match(1) | df['QUEMADO'].str.match(1)]
		ImageFile.LOAD_TRUNCATED_IMAGES = True
		
		# Eliminar los elementos que no contengan una imagen
		eliminar =  set() # Se crea un conjunto de los indices a eliminar
		df_si.reset_index(drop=True, inplace=True) # Se utiliza el indice que ofrece pandas por defecto
		for index, row in df_s.iterrows(): # Por cada columna
			if not row['FOTO'].is_file():
				eliminar.add(index) # Si el archivo de la foto no existe se ayade su indice a la lista de eliminar
		sobreviven = set(range(df_s.shape[0])) - eliminar # Se restan todos los indices con los que se eliminan para crear los que sobreviven
		df_si = df_s.take(list(sobreviven)) # Se sobrescribe el df solo con las filas con indices que sobreviven 
		
		train, test, valid =  crear_conjuntos(df_si.values)
		
		#Datos del entrenamiento
		epocas_sin_liberar = 10
		epocas_con_liberar = 10
		batch_size = 32
		learning_rate = 0.0001
		num_clases = 3
		
		modelo = Resnext50(block=True, num_clases)
		optimizador = torch.optim.Adam(modelo.parameters(), lr = learning_rate)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		#################### Experimento: 1 #######################
		predida = nn.BCELoss()
		
		train_data = ContenedorDataset(train, transform=transforms.Compose([transforms.Resize((400,400)),transforms.ToTensor()]))
		valid_data = ContenedorDataset(valid, transform=transforms.Compose([transforms.Resize((400,400)),transforms.ToTensor()]))
		test_data = ContenedorDataset(test, transform=transforms.Compose([transforms.Resize((400,400)),transforms.ToTensor()])) 
		
		
		#################### Experimento: 2 #######################
		predida = MarginLoss()
		
		train_data = ContenedorDataset(train, transform=transforms.Compose([transforms.Resize((400,400)),transforms.ToTensor()]))
		valid_data = ContenedorDataset(valid, transform=transforms.Compose([transforms.Resize((400,400)),transforms.ToTensor()]))
		test_data = ContenedorDataset(test, transform=transforms.Compose([transforms.Resize((400,400)),transforms.ToTensor()])) 
		
		
		#################### Experimento: 3 #######################
		predida = nn.BCELoss()
		
		train_data = ContenedorDataset(train, transform=transforms.Compose([
			transforms.Resize((400,400)),
			transforms.RandomCrop(350),
			transforms.Resize((400,400)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor()]))
		valid_data = ContenedorDataset(valid, transform=transforms.Compose([transforms.Resize((400,400)),transforms.ToTensor()]))
		test_data = ContenedorDataset(test, transform=transforms.Compose([transforms.Resize((400,400)),transforms.ToTensor()])) 
		
		#Cargadores de datos
		carga_train = DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True)
		carga_valid = DataLoader(dataset = valid_data, batch_size = batch_size, shuffle=False)
		carga_test = DataLoader(dataset = test_data, batch_size = 1, shuffle=False)
		
		# Entrenamiento del modelo pre entrenado
		train(optimizador, modelo, perdida, epocas_sin_liberar, carga_train, carga_valid, "SinLiberar")
		guardarModelo(modelo, optimizador, "ModeloSinLiberar.ckpt")
		modelo = Resnext50()
		optimizador = torch.optim.Adam(modelo.parameters(),lr = learning_rate)
		checkpoint = torch.load("ModeloSinLiberar.ckpt")
		modelo.load_state_dict(checkpoint['model_state_dict'])
		optimizador.load_state_dict(checkpoint['optimizer_state_dict'])
		train(optimizador, modelo, perdida, epocas_con_liberar, carga_train, carga_valid, "Completo")
		
		#Test y guarda el modelo
		soterrado_predict, grafiti_predict, quemado_predict = test(modelo, carga_test)
		guardarModelo(modelo, optimizador, "ModeloCompleto.ckpt")
