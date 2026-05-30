## Primeri za lab4
1. Koristeći CUDA tehnologiju, napisati program koji za dati niz An računa niz Bn-2 na sledeći način:
B[i] = (A[i] + A[i+1] + A[i+2]) / 3 
Maksimalno redukovati broj pristupa globalnoj memoriji. Obratiti pažnju na efikasnost paralelizacije. Omogućiti rad programa za matrice proizvoljne veličine.

2. Koristeći CUDA tehnologiju, sastaviti program koji na osnovu matrice Anxn kreira vektor Bn, gde je element B[i] jednak sumi elemenata i-te kolone matrice A. Maksimalno redukovati broj pristupa globalnoj memoriji. Obratiti pažnju na efikasnost paralelizacije. Omogućiti rad programa za matrice proizvoljne veličine.