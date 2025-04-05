import pandas as pd
import numpy as np
import re
import unicodedata
from typing import List, Dict, Tuple, Optional, Union, Set

class TextNormalizer:
    """Clase para normalizar textos y aplicar transformaciones comunes."""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normaliza un texto: convierte a minúsculas, elimina acentos y caracteres especiales,
        preservando espacios.
        
        Args:
            text (str): Texto a normalizar
            
        Returns:
            str: Texto normalizado
        """
        if not isinstance(text, str):
            return text
            
        # Convertir a minúsculas
        text = text.lower()
        
        # Normalización NFKD para separar caracteres base y diacríticos
        normalized_text = unicodedata.normalize('NFKD', text)
        
        # Eliminar diacríticos y mantener solo caracteres alfanuméricos y espacios
        clean_text = re.sub(r'[^a-z0-9 ]', '', normalized_text)
        
        return clean_text


class NameSeparator:
    """Clase para separar nombres de proyectos y personas."""
    
    def __init__(self, projects_list: List[str]):
        """
        Inicializa el separador con una lista de proyectos conocidos.
        
        Args:
            projects_list (List[str]): Lista de nombres de proyectos normalizados
        """
        self.projects_list = projects_list
        
    def separate_names(self, business_name: str, normalized_business_name: str, 
                      project_name_mapping: Dict[str, str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Separa el nombre del negocio en nombre de persona y nombre de proyecto.
        
        Args:
            business_name (str): Nombre del negocio original
            normalized_business_name (str): Nombre del negocio normalizado
            project_name_mapping (Dict[str, str]): Mapeo de nombres de proyectos normalizados a originales
            
        Returns:
            Tuple[Optional[str], Optional[str]]: Tupla con (nombre de persona, nombre de proyecto)
        """
        if not isinstance(business_name, str) or not isinstance(normalized_business_name, str):
            return None, None
        
        # Buscar si alguno de los proyectos conocidos está en el texto normalizado
        project_found = None
        for project in self.projects_list:
            if project in normalized_business_name:
                project_found = project
                break
        
        # Si se encontró un proyecto
        if project_found:
            # Encontrar el nombre del proyecto original correspondiente
            project_original = project_found.upper()
            if project_name_mapping and project_found in project_name_mapping:
                project_original = project_name_mapping[project_found]
            
            # Intentar dividir el texto original
            if '-' in business_name:
                parts = [part.strip() for part in business_name.split('-', 1)]
                
                # Ver cuál parte contiene el proyecto
                if TextNormalizer.normalize_text(parts[0]) == project_found or project_found in TextNormalizer.normalize_text(parts[0]):
                    return parts[1], project_original
                else:
                    return parts[0], project_original
            else:
                # Si no hay guión, extraer la parte que no es el proyecto
                rest = business_name.replace(project_original, '').strip()
                if rest:
                    return rest, project_original
                else:
                    return None, project_original
        else:
            # Si no se encuentra ningún proyecto conocido
            if '-' in business_name:
                parts = [part.strip() for part in business_name.split('-', 1)]
                return parts[1], parts[0]
            else:
                return business_name, None


class SpamDetector:
    """Clase para detectar nombres spam usando reglas heurísticas."""
    
    def __init__(self, projects_list: Optional[List[str]] = None, spam_words: Optional[List[str]] = None):
        """
        Inicializa el detector de spam.
        
        Args:
            projects_list (Optional[List[str]]): Lista opcional de nombres de proyectos normalizados
            spam_words (Optional[List[str]]): Lista personalizada de palabras spam
        """
        self.projects_list = projects_list or []
        
        # Si no se proporciona una lista personalizada, usar la predeterminada
        if spam_words is None:
            self.spam_words = [
                'user', 'test', 'temp', 'anon', 'info', 'www', 'mail', 'promo', 'click', 
                'admin', 'guest', 'anonymous', 'system', 'none', 'default', 'demo', 'cliente',
                'apartamento', 'universidad', 'empresa', 'negocio', 'proyecto', 'nuevo negocio'
            ]
        else:
            self.spam_words = spam_words
        
    def detect_spam(self, name: str) -> Tuple[bool, str]:
        """
        Detecta si un nombre es spam basado en reglas heurísticas.
        
        Args:
            name (str): Nombre normalizado a analizar
            
        Returns:
            Tuple[bool, str]: (Es spam (True/False), Regla que activó la detección)
        """
        # REGLA NUEVA: Nombres vacíos o nulos son spam
        if name is None or not isinstance(name, str) or pd.isna(name) or name.strip() == '':
            return True, "Nombre vacío o nulo"
            
        # REGLA 1: Nombres extremadamente cortos (menos de 4 caracteres)
        # Ajustada para reducir falsos positivos con nombres reales
        if len(name) < 4:
            return True, "Nombre extremadamente corto"
            
        # REGLA 2: Nombres con muchos dígitos
        digits = sum(c.isdigit() for c in name)
        if digits > 0:
            proportion_digits = digits / len(name)
            if proportion_digits > 0.2:  # Si más del 20% son dígitos
                return True, "Alta proporción de dígitos"
                
        # REGLA 3: Nombres que contienen palabras sospechosas
        if any(word in name.lower() for word in self.spam_words):
            return True, "Contiene palabra sospechosa"
            
        # REGLA 4: Nombres con caracteres repetidos extrañamente
        if re.search(r'(.)\1{3,}', name):  # 3 o más caracteres iguales seguidos
            return True, "Caracteres repetidos"
        
        # REGLA 5: Nombres con patrones de repetición (como aabbcc)
        if re.match(r'^(.)\1{1,}([a-z0-9])\2{1,}$', name):
            return True, "Patrón de repetición"
        
        # REGLA 6: Nombres con patrones extraños de alternancia (como a1b2c3)
        if re.match(r'^([a-z][0-9])+$', name) or re.match(r'^([0-9][a-z])+$', name):
            return True, "Patrón de alternancia"
            
        # REGLA 7: Nombres sin vocales
        if len(name) > 4 and not any(c in 'aeiou' for c in name.lower()):
            return True, "Sin vocales"
            
        # REGLA 8: Nombres con muchos guiones bajos seguidos
        if '__' in name:
            return True, "Guiones bajos múltiples"
            
        # REGLA 9: Nombres sin sentido (extremadamente pocas vocales)
        # Ajustada para considerar apellidos con menor proporción de vocales
        if len(name) > 5:
            vowels = sum(c.lower() in 'aeiou' for c in name)
            vowel_proportion = vowels / len(name)
            if vowel_proportion < 0.10:  # Reducido de 0.15 a 0.10
                return True, "Muy pocas vocales"
                
        # REGLA 10: Nombres que contengan al menos un número
        if any(c.isdigit() for c in name):
            return True, "Contiene dígitos"
            
        # REGLA 11: Nombres que tengan más de 50 caracteres de longitud
        if len(name) > 50:
            return True, "Nombre extremadamente largo"
            
        # REGLA 12: Coincidencia con nombres de proyectos
        if self.projects_list:
            name_lower = name.lower()
            # Coincidencia exacta
            if name_lower in self.projects_list:
                return True, "Coincide con nombre de proyecto"
            
            # Coincidencia parcial significativa
            for project in self.projects_list:
                if project and len(project) > 3 and project in name_lower:
                    return True, "Contiene nombre de proyecto"
        
        # Verificar si hay múltiples palabras y analizar cada una
        if ' ' in name:
            words = name.split()
            # Nombres reales suelen tener 2-3 palabras (nombre y apellidos)
            if 2 <= len(words) <= 4:
                # Verificar si cada parte tiene una longitud razonable
                if all(len(word) >= 3 for word in words):
                    return False, "Nombre válido con múltiples palabras"
                    
        return False, "No es spam"


class SpamAnalyzer:
    """Clase principal para analizar datos y detectar nombres spam."""
    
    def __init__(self, spam_words: Optional[List[str]] = None):
        """
        Inicializa el analizador de spam.
        
        Args:
            spam_words (Optional[List[str]]): Lista personalizada de palabras spam
        """
        self.text_normalizer = TextNormalizer()
        self.spam_words = spam_words
        
    def analyze_data(self, 
                  df_datos: pd.DataFrame, 
                  df_proyectos: pd.DataFrame,
                  col_negocio: str = 'Nombre del negocio',
                  col_proyecto_datos: str = 'Nombre del proyecto',
                  col_proyecto_lista: str = 'PROYECTOS AÑO 2025') -> Tuple[pd.DataFrame, Dict]:
        """
        Analiza los datos para detectar nombres spam en un solo paso.
        
        Args:
            df_datos (pd.DataFrame): DataFrame con datos principales
            df_proyectos (pd.DataFrame): DataFrame con lista de proyectos
            col_negocio (str): Nombre de la columna con nombres de negocios
            col_proyecto_datos (str): Nombre de la columna de proyectos en df_datos
            col_proyecto_lista (str): Nombre de la columna de proyectos en df_proyectos
            
        Returns:
            Tuple[pd.DataFrame, Dict]: DataFrame procesado y estadísticas de spam
        """
        # 1. Normalizar nombres de proyectos en ambos DataFrames
        df_datos = df_datos.copy()
        df_proyectos = df_proyectos.copy()
        
        col_proyecto_norm = 'nombre_proyecto_normalizado'
        col_negocio_norm = f"{col_negocio}_normalizado"
        
        # Normalizar proyectos en la lista
        df_proyectos[col_proyecto_norm] = df_proyectos[col_proyecto_lista].apply(
            self.text_normalizer.normalize_text
        )
        
        # Normalizar proyectos en los datos
        df_datos[col_proyecto_norm] = df_datos[col_proyecto_datos].apply(
            self.text_normalizer.normalize_text
        )
        
        # 2. Filtrar datos con proyectos conocidos
        proyectos_lista = df_proyectos[col_proyecto_norm].dropna().unique().tolist()
        df_filtrado = df_datos[df_datos[col_proyecto_norm].isin(proyectos_lista)].copy()
        
        # Si no hay datos después del filtrado, devolver DataFrame vacío
        if len(df_filtrado) == 0:
            return df_filtrado, {'spam_count': 0, 'total_count': 0, 'spam_percentage': 0}
        
        # 3. Normalizar nombres de negocios
        df_filtrado[col_negocio_norm] = df_filtrado[col_negocio].apply(
            self.text_normalizer.normalize_text
        )
        
        # 4. Crear mapeo de proyectos normalizados a originales
        project_mapping = {}
        for _, row in df_filtrado.drop_duplicates([col_proyecto_norm, col_proyecto_datos]).iterrows():
            if pd.notna(row[col_proyecto_norm]) and pd.notna(row[col_proyecto_datos]):
                project_mapping[row[col_proyecto_norm]] = row[col_proyecto_datos]
        
        # 5. Separar nombres de personas y proyectos
        name_separator = NameSeparator(proyectos_lista)
        
        # Aplicar la separación a cada fila
        person_project_pairs = []
        for _, row in df_filtrado.iterrows():
            person, project = name_separator.separate_names(
                row[col_negocio],
                row[col_negocio_norm],
                project_mapping
            )
            person_project_pairs.append((person, project))
        
        # Añadir las nuevas columnas al DataFrame
        df_filtrado['Nombre_de_la_persona'] = [pair[0] for pair in person_project_pairs]
        df_filtrado['Nombre_del_proyecto_final'] = [pair[1] for pair in person_project_pairs]
        
        # 6. Normalizar los nombres de personas
        df_filtrado['Nombre_de_la_persona_normalizado'] = df_filtrado['Nombre_de_la_persona'].apply(
            self.text_normalizer.normalize_text
        )
        
        # Hacer un post-procesamiento para manejar nombres compuestos
        df_filtrado['Nombre_de_la_persona_normalizado'] = df_filtrado['Nombre_de_la_persona_normalizado'].apply(
            lambda x: x if not isinstance(x, str) else x.strip()
        )
        
        # 7. Detectar nombres spam
        spam_detector = SpamDetector(proyectos_lista, self.spam_words)
        
        # Aplicar detección y capturar tanto el resultado como la razón
        spam_results = df_filtrado['Nombre_de_la_persona_normalizado'].apply(
            lambda name: spam_detector.detect_spam(name)
        )
        
        # Separar los resultados en columnas separadas
        df_filtrado['Spam'] = [result[0] for result in spam_results]
        df_filtrado['Razon_Spam'] = [result[1] for result in spam_results]
        
        # Convertir boolean a int para la columna Spam
        df_filtrado['Spam'] = df_filtrado['Spam'].astype(int)
        
        # 8. Obtener estadísticas
        spam_count = df_filtrado['Spam'].sum()
        total_count = len(df_filtrado)
        spam_percentage = (spam_count / total_count) * 100 if total_count > 0 else 0
        
        # Ejemplos de nombres clasificados como spam/normales
        spam_examples = df_filtrado[df_filtrado['Spam'] == 1]['Nombre_de_la_persona_normalizado'].head(10).tolist()
        normal_examples = df_filtrado[df_filtrado['Spam'] == 0]['Nombre_de_la_persona_normalizado'].head(10).tolist()
        
        stats = {
            'spam_count': spam_count,
            'total_count': total_count,
            'spam_percentage': spam_percentage,

        }
        
        return df_filtrado, stats