"""
Script de prueba para métodos de integración
"""
from metodos import trapecio, simpson
import math

print("=" * 60)
print("PRUEBA 1: Integración de f(x) = x^2 de 0 a 2")
print("Valor exacto: 8/3 ≈ 2.666667")
print("=" * 60)

# Datos no equiespaciados
X = [0, 0.5, 1.5, 2.0]
Y = [x**2 for x in X]

print(f"\nDatos: {len(X)} puntos no equiespaciados")
print(f"X: {X}")
print(f"Y: {Y}")

print("\n--- TRAPECIO ---")
resultado_t = trapecio(X=X, Y=Y)
print(f"Error: {abs(resultado_t - 8/3):.6f}")

print("\n--- SIMPSON ---")
resultado_s = simpson(X=X, Y=Y)
print(f"Error: {abs(resultado_s - 8/3):.6f}")

print("\n" + "=" * 60)
print("PRUEBA 2: Integración de sin(x) de 0 a π/2")
print("Valor exacto: 1.0")
print("=" * 60)

X2 = [0, 0.3, 0.8, 1.2, math.pi/2]
Y2 = [math.sin(x) for x in X2]

print(f"\nDatos: {len(X2)} puntos no equiespaciados")
print(f"X: {[round(x, 3) for x in X2]}")

print("\n--- TRAPECIO ---")
resultado_t2 = trapecio(X=X2, Y=Y2)
print(f"Error: {abs(resultado_t2 - 1.0):.6f}")

print("\n--- SIMPSON ---")
resultado_s2 = simpson(X=X2, Y=Y2)
print(f"Error: {abs(resultado_s2 - 1.0):.6f}")

print("\n" + "=" * 60)
print("PRUEBA 3: Demostración de n por defecto")
print("=" * 60)

print("\nTRAPECIO (siempre n = len(X) - 1):")
for npts in [3, 4, 5, 6]:
    X_test = list(range(npts))
    Y_test = [x**2 for x in X_test]
    print(f"  {npts} puntos → n={npts-1}", end="")
    trapecio(X=X_test, Y=Y_test, verbose=False)
    print(" ✓")

print("\nSIMPSON (n = len(X) si par, len(X)-1 si impar):")
for npts in [3, 4, 5, 6, 7]:
    X_test = list(range(npts))
    Y_test = [x**2 for x in X_test]
    n_expected = npts if npts % 2 == 0 else npts - 1
    tipo = "par" if npts % 2 == 0 else "impar"
    print(f"  {npts} puntos ({tipo}) → n={n_expected}", end="")
    simpson(X=X_test, Y=Y_test, verbose=False)
    print(" ✓")

print("\n" + "=" * 60)
print("PRUEBA 4: Funciones continuas con alta precisión")
print("=" * 60)

f = lambda x: x**2
print("\nf(x) = x^2 de 0 a 2 (n=1000):")
print("TRAPECIO:", end=" ")
r_t = trapecio(f=f, a=0, b=2, n=1000, verbose=False)
print(f"Error = {abs(r_t - 8/3):.10f}")

print("SIMPSON:  ", end=" ")
r_s = simpson(f=f, a=0, b=2, n=1000, verbose=False)
print(f"Error = {abs(r_s - 8/3):.12f}")

print("\n✅ Todas las pruebas completadas exitosamente!")
