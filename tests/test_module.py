import unittest
from nom_de_votre_package import module  # Ajustez le chemin d'importation selon votre structure

class TestModule(unittest.TestCase):
    def test_fonction1(self):
        # Supposons que module.fonction1() doive retourner True
        self.assertTrue(module.fonction1())

    def test_fonction2_avec_parametre(self):
        # Supposons que module.fonction2(5) doive retourner 10
        resultat_attendu = 10
        self.assertEqual(module.fonction2(5), resultat_attendu)

# Ceci permet d'exécuter les tests si ce fichier est exécuté directement
if __name__ == '__main__':
    unittest.main()