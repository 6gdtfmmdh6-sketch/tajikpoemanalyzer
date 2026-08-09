#!/usr/bin/env python3
"""
MARC21/XML Export für bibliothekarische Integration
"""
from lxml import etree
from dataclasses import asdict
from datetime import datetime

class MARCExtractor:
    """MARC21 XML Export für Poesie-Bände"""
    
    NAMESPACES = {
        'marc': 'http://www.loc.gov/MARC21/slim'
    }
    
    def create_marc_record(self, volume_metadata, poems_data) -> str:
        """Erstellt MARC21 XML Record"""
        
        root = etree.Element('record', xmlns=self.NAMESPACES['marc'])
        
        # Control Field 001 (Identifier)
        control_field = etree.SubElement(root, 'controlfield', tag='001')
        control_field.text = f"TAJIK-POETRY-{datetime.now().timestamp()}"
        
        # Datenfelder entsprechend MARC21
        # 100: Haupteingang (Autor)
        author_field = etree.SubElement(root, 'datafield', tag='100', ind1='1', ind2=' ')
        etree.SubElement(author_field, 'subfield', code='a').text = volume_metadata.author_name
        
        # 245: Titel
        title_field = etree.SubElement(root, 'datafield', tag='245', ind1='1', ind2='0')
        etree.SubElement(title_field, 'subfield', code='a').text = volume_metadata.volume_title
        etree.SubElement(title_field, 'subfield', code='c').text = volume_metadata.author_name
        
        # 260: Veröffentlichungsinformation
        pub_field = etree.SubElement(root, 'datafield', tag='260', ind1=' ', ind2=' ')
        etree.SubElement(pub_field, 'subfield', code='c').text = str(volume_metadata.publication_year)
        
        # 653: Indexbegriff (Poesie-Analyse)
        index_field = etree.SubElement(root, 'datafield', tag='653', ind1=' ', ind2=' ')
        etree.SubElement(index_field, 'subfield', code='a').text = "Tajik poetry"
        etree.SubElement(index_field, 'subfield', code='a').text = "ʿArūḍ analysis"
        
        return etree.tostring(root, pretty_print=True, encoding='unicode')
