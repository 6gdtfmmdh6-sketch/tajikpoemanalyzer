#!/usr/bin/env python3
"""
Comprehensive Testing and Validation Suite for Tajik Poetry Analyzer

This module provides:
1. Unit tests for all components
2. Integration tests
3. Scientific validation protocols
4. Performance benchmarks
5. Error analysis and debugging tools

Usage:
    python test_validation_suite.py --run-all
    python test_validation_suite.py --test phonetic
    python test_validation_suite.py --validate results.json
"""

import sys
import json
import time
import argparse
import unittest
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhoneticAnalysisTests(unittest.TestCase):
    """Unit tests for phonetic analysis components"""

    def setUp(self):
        from phonetic_utils import PersianPhonology
        self.phonology = PersianPhonology()

    def test_basic_transcription(self):
        """Test basic phonetic transcription"""
        test_cases = [
            ("муҳаббат", "muħabbat"),
            ("дарё", "darjo"),
            ("кӯҳ", "kuːh"),
            ("баҳор", "baħor"),
            ("гул", "gul")
        ]

        for original, expected in test_cases:
            with self.subTest(word=original):
                phonetic, confidence = self.phonology.enhanced_phonetic_transcription(original)
                self.assertIsInstance(phonetic, str)
                self.assertGreater(confidence, 0.5)
                # Basic check - should contain expected sounds
                self.assertIn(expected[0], phonetic)  # First sound should match

    def test_syllable_boundaries(self):
        """Test syllable boundary detection"""
        test_cases = [
            ("muħabbat", 3),  # mu-ħab-bat
            ("darjo", 2),  # dar-jo
            ("kuːh", 1),  # kuːh
            ("baħor", 2),  # ba-ħor
            ("gul", 1)  # gul
        ]

        for phonetic, expected_syllables in test_cases:
            with self.subTest(phonetic=phonetic):
                boundaries = self.phonology.find_syllable_boundaries_advanced(phonetic)
                actual_syllables = len(boundaries) - 1
                self.assertEqual(actual_syllables, expected_syllables,
                                 f"Expected {expected_syllables} syllables in {phonetic}, got {actual_syllables}")

    def test_syllable_weight_calculation(self):
        """Test syllable weight calculation"""
        test_cases = [
            ("kuːh", "final", "—"),  # Long vowel -> heavy
            ("ba", "medial", "U"),  # CV -> light
            ("ħab", "medial", "—"),  # CVC -> heavy
            ("gul", "final", "—"),  # CVC final -> heavy
        ]

        for syllable, position, expected_weight in test_cases:
            with self.subTest(syllable=syllable):
                weight = self.phonology.calculate_syllable_weight_advanced(syllable, position)
                self.assertEqual(weight, expected_weight,
                                 f"Expected weight {expected_weight} for {syllable}, got {weight}")


class AruzMeterTests(unittest.TestCase):
    """Unit tests for ʿArūḍ meter analysis"""

    def setUp(self):
        from enhanced_tajik_analyzer import AruzMeterAnalyzer
        self.analyzer = AruzMeterAnalyzer()

    def test_meter_detection(self):
        """Test meter detection on known examples"""
        # Note: These are simplified test cases
        # Real validation would require expert-annotated corpus
        test_lines = [
            "Дар кӯҳсори ватан гулҳо мешуканд",
            "Дили шоир аз муҳаббат меларазад",
            "Баҳори нав ба замин таҷдид меорад"
        ]

        for line in test_lines:
            with self.subTest(line=line[:20] + "..."):
                analysis = self.analyzer.analyze_meter(line)

                # Basic checks
                self.assertIsNotNone(analysis.identified_meter)
                self.assertIsInstance(analysis.pattern_match, str)
                self.assertIsInstance(analysis.pattern_accuracy, float)
                self.assertGreaterEqual(analysis.pattern_accuracy, 0.0)
                self.assertLessEqual(analysis.pattern_accuracy, 1.0)

    def test_pattern_similarity(self):
        """Test prosodic pattern similarity calculation"""
        test_cases = [
            ("U—U—", "U—U—", 1.0),  # Identical
            ("U—U—", "U—UU", 0.75),  # Similar
            ("U—U—", "—U—U", 0.5),  # Different
            ("", "", 1.0),  # Empty
            ("U", "—", 0.0),  # Completely different
        ]

        for pattern1, pattern2, expected_min_similarity in test_cases:
            with self.subTest(p1=pattern1, p2=pattern2):
                similarity = self.analyzer._pattern_similarity(pattern1, pattern2)
                self.assertGreaterEqual(similarity, expected_min_similarity - 0.1)
                self.assertLessEqual(similarity, 1.0)


class RhymeAnalysisTests(unittest.TestCase):
    """Unit tests for rhyme analysis"""

    def setUp(self):
        from enhanced_tajik_analyzer import AdvancedRhymeAnalyzer
        self.analyzer = AdvancedRhymeAnalyzer()

    def test_qafiyeh_extraction(self):
        """Test qāfiyeh extraction"""
        test_cases = [
            ("Дар кӯҳсори ватан гулҳо мешуканд", "анд"),
            ("Дили шоир аз муҳаббат меларазад", "зад"),
            ("Баҳори нав ба замин таҷдид меорад", "рад")
        ]

        for line, expected_ending in test_cases:
            with self.subTest(line=line[:20] + "..."):
                analysis = self.analyzer.analyze_rhyme(line)
                self.assertIsInstance(analysis.qafiyeh, str)
                # Should contain expected ending
                self.assertTrue(analysis.qafiyeh.endswith(expected_ending[-2:]))

    def test_radif_detection(self):
        """Test radīf detection"""
        # Lines with repeated refrains
        test_lines = [
            "شابات хуш боду рӯзат хуш",  # "хуш" repeated
            "дар ин ҷаҳон зебо аст",  # No radīf
        ]

        for line in test_lines:
            with self.subTest(line=line):
                analysis = self.analyzer.analyze_rhyme(line)
                self.assertIsInstance(analysis.radif, str)
                # Radīf should be shorter than the line
                self.assertLess(len(analysis.radif), len(line))


class IntegrationTests(unittest.TestCase):
    """Integration tests for complete analysis pipeline"""

    def setUp(self):
        from enhanced_tajik_analyzer import EnhancedTajikPoemAnalyzer
        self.analyzer = EnhancedTajikPoemAnalyzer()

    def test_complete_poem_analysis(self):
        """Test complete analysis of a poem"""
        sample_poem = """
        Дар кӯҳсори ватан гулҳо мешуканд,
        Дили шоир аз муҳаббат меларазад.

        Баҳори нав ба замин таҷдид меорад,
        Навиди хушҳолии мардум мерасад.
        """

        try:
            result = self.analyzer.analyze_poem_enhanced(sample_poem)

            # Check structure
            self.assertIn('structural_analysis', result)
            self.assertIn('validation', result)
            self.assertIn('metadata', result)

            # Check structural analysis
            structural = result['structural_analysis']
            self.assertGreater(structural.lines, 0)
            self.assertIsInstance(structural.avg_syllables, float)
            self.assertGreater(structural.avg_syllables, 0)

            # Check validation
            validation = result['validation']
            self.assertIn('quality_score', validation)
            self.assertIn('reliability_level', validation)
            self.assertIsInstance(validation['quality_score'], float)

        except Exception as e:
            self.fail(f"Complete analysis failed: {e}")

    def test_error_handling(self):
        """Test error handling with problematic input"""
        problematic_inputs = [
            "",  # Empty
            "   ",  # Whitespace only
            "123 456 789",  # Numbers only
            "!@#$%^&*()",  # Punctuation only
            "A" * 1000,  # Very long
        ]

        for problematic_input in problematic_inputs:
            with self.subTest(input=problematic_input[:20] + "..."):
                try:
                    result = self.analyzer.analyze_poem_enhanced(problematic_input)
                    # Should not crash, but may have low quality scores
                    self.assertIsInstance(result, dict)
                except ValueError:
                    # ValueError is acceptable for clearly invalid input
                    pass
                except Exception as e:
                    self.fail(f"Unexpected error for input '{problematic_input[:20]}...': {e}")


class PerformanceBenchmarks:
    """Performance benchmarking for scientific applications"""

    def __init__(self):
        from enhanced_tajik_analyzer import EnhancedTajikPoemAnalyzer
        self.analyzer = EnhancedTajikPoemAnalyzer()

    def benchmark_analysis_speed(self, test_poems: List[str]) -> Dict[str, float]:
        """Benchmark analysis speed"""
        results = {
            "total_poems": len(test_poems),
            "total_time": 0.0,
            "avg_time_per_poem": 0.0,
            "poems_per_second": 0.0,
            "individual_times": []
        }

        start_time = time.time()

        for i, poem in enumerate(test_poems):
            poem_start = time.time()
            try:
                self.analyzer.analyze_poem_enhanced(poem)
                poem_time = time.time() - poem_start
                results["individual_times"].append(poem_time)
            except Exception as e:
                logger.warning(f"Poem {i} failed analysis: {e}")
                results["individual_times"].append(None)

        total_time = time.time() - start_time
        valid_times = [t for t in results["individual_times"] if t is not None]

        results["total_time"] = total_time
        results["avg_time_per_poem"] = sum(valid_times) / len(valid_times) if valid_times else 0
        results["poems_per_second"] = len(valid_times) / total_time if total_time > 0 else 0

        return results

    def benchmark_memory_usage(self, test_poems: List[str]) -> Dict[str, Any]:
        """Benchmark memory usage (simplified)"""
        # Note: For production, would use memory_profiler or similar
        import sys

        initial_objects = len(gc.get_objects()) if 'gc' in sys.modules else 0

        for poem in test_poems:
            try:
                self.analyzer.analyze_poem_enhanced(poem)
            except Exception:
                pass

        final_objects = len(gc.get_objects()) if 'gc' in sys.modules else 0

        return {
            "initial_objects": initial_objects,
            "final_objects": final_objects,
            "object_growth": final_objects - initial_objects
        }


class ScientificValidation:
    """Scientific validation protocols"""

    @staticmethod
    def validate_against_expert_annotations(analysis_results: Dict[str, Any],
                                            expert_annotations: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate analysis against expert annotations

        This would be used with a gold standard corpus annotated by prosody experts.
        """
        validation_scores = {
            "meter_accuracy": 0.0,
            "rhyme_accuracy": 0.0,
            "syllable_accuracy": 0.0,
            "overall_accuracy": 0.0
        }

        # This is a simplified implementation
        # Real validation would require detailed comparison metrics

        logger.info("Expert validation requires gold standard corpus - not implemented in demo")
        return validation_scores

    @staticmethod
    def statistical_analysis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform statistical analysis of results
        """
        if not results:
            return {"error": "No results to analyze"}

        # Extract metrics
        quality_scores = [r.get('validation', {}).get('quality_score', 0) for r in results]
        meter_confidences = []
        rhyme_confidences = []

        for result in results:
            structural = result.get('structural_analysis', {})
            if hasattr(structural, 'meter_confidence'):
                # Convert enum to numeric for analysis
                confidence_map = {'high': 0.9, 'medium': 0.7, 'low': 0.5, 'none': 0.1}
                meter_confidences.append(confidence_map.get(structural.meter_confidence.value, 0.1))

            if hasattr(structural, 'rhyme_scheme'):
                rhyme_conf = [r.confidence for r in structural.rhyme_scheme if hasattr(r, 'confidence')]
                if rhyme_conf:
                    rhyme_confidences.append(sum(rhyme_conf) / len(rhyme_conf))

        # Calculate statistics
        def safe_stats(values):
            if not values:
                return {"mean": 0, "std": 0, "min": 0, "max": 0}
            return {
                "mean": sum(values) / len(values),
                "std": (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)) ** 0.5,
                "min": min(values),
                "max": max(values)
            }

        return {
            "total_poems": len(results),
            "quality_scores": safe_stats(quality_scores),
            "meter_confidences": safe_stats(meter_confidences),
            "rhyme_confidences": safe_stats(rhyme_confidences),
            "successful_analyses": sum(1 for r in results if r.get('validation', {}).get('quality_score', 0) > 0.5)
        }


def generate_test_poems() -> List[str]:
    """Generate test poems for benchmarking"""
    return [
        """
        Дар кӯҳсори ватан гулҳо мешуканд,
        Дили шоир аз муҳаббат меларазад.
        """,
        """
        Баҳори нав ба замин таҷдид меорад,
        Навиди хушҳолии мардум мерасад.
        """,
        """
        Дуруд, эй қосиди фардо, шабат хуш боду рӯзат хуш,
        Навиншоиву ноифшо, шабат хуш боду рӯзат хуш.
        """,
        """
        Видоъ, эй рафтаи ширин, шабат хуш буду рӯзат хуш,
        Дуруд ояндаи зебо, шабат хуш боду рӯзат хуш.
        """,
        """
        Чӣ мерақсӣ даруни хонақоҳи сина рӯзу шаб,
        Дил, эй дил, солики танҳо, шабат хуш боду рӯзат хуш.
        """
    ]


def run_comprehensive_tests():
    """Run all test suites"""
    print("=== COMPREHENSIVE TESTING SUITE ===\n")

    # Unit tests
    print("Running unit tests...")
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    # Add test classes
    test_suite.addTests(test_loader.loadTestsFromTestCase(PhoneticAnalysisTests))
    test_suite.addTests(test_loader.loadTestsFromTestCase(AruzMeterTests))
    test_suite.addTests(test_loader.loadTestsFromTestCase(RhymeAnalysisTests))
    test_suite.addTests(test_loader.loadTestsFromTestCase(IntegrationTests))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(test_suite)

    print(f"\nUnit tests completed:")
    print(f"  Tests run: {test_result.testsRun}")
    print(f"  Failures: {len(test_result.failures)}")
    print(f"  Errors: {len(test_result.errors)}")

    return test_result.wasSuccessful()


def run_performance_benchmarks():
    """Run performance benchmarks"""
    print("\n=== PERFORMANCE BENCHMARKS ===\n")

    test_poems = generate_test_poems()
    benchmarker = PerformanceBenchmarks()

    # Speed benchmark
    print("Running speed benchmark...")
    speed_results = benchmarker.benchmark_analysis_speed(test_poems)

    print(f"Speed Results:")
    print(f"  Total poems: {speed_results['total_poems']}")
    print(f"  Total time: {speed_results['total_time']:.2f} seconds")
    print(f"  Average time per poem: {speed_results['avg_time_per_poem']:.3f} seconds")
    print(f"  Poems per second: {speed_results['poems_per_second']:.2f}")

    # Memory benchmark
    print("\nRunning memory benchmark...")
    memory_results = benchmarker.benchmark_memory_usage(test_poems)

    print(f"Memory Results:")
    print(f"  Object growth: {memory_results['object_growth']}")

    return speed_results, memory_results


def run_scientific_validation(results_file: str = None):
    """Run scientific validation protocols"""
    print("\n=== SCIENTIFIC VALIDATION ===\n")

    # Generate sample results if no file provided
    if not results_file:
        print("Generating sample analysis results for validation...")
        from enhanced_tajik_analyzer import EnhancedTajikPoemAnalyzer

        analyzer = EnhancedTajikPoemAnalyzer()
        test_poems = generate_test_poems()

        results = []
        for i, poem in enumerate(test_poems):
            try:
                result = analyzer.analyze_poem_enhanced(poem)
                results.append(result)
            except Exception as e:
                print(f"Failed to analyze poem {i}: {e}")

    else:
        # Load results from file
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception as e:
            print(f"Failed to load results file: {e}")
            return

    # Statistical analysis
    validator = ScientificValidation()
    stats = validator.statistical_analysis(results)

    print("Statistical Analysis:")
    print(f"  Total poems analyzed: {stats['total_poems']}")
    print(f"  Successful analyses: {stats['successful_analyses']}")
    print(f"  Success rate: {stats['successful_analyses'] / stats['total_poems'] * 100:.1f}%")

    print(f"\nQuality Score Statistics:")
    qs = stats['quality_scores']
    print(f"  Mean: {qs['mean']:.3f}")
    print(f"  Std Dev: {qs['std']:.3f}")
    print(f"  Range: {qs['min']:.3f} - {qs['max']:.3f}")

    print(f"\nMeter Confidence Statistics:")
    mc = stats['meter_confidences']
    print(f"  Mean: {mc['mean']:.3f}")
    print(f"  Std Dev: {mc['std']:.3f}")
    print(f"  Range: {mc['min']:.3f} - {mc['max']:.3f}")

    # Quality assessment
    overall_quality = qs['mean']
    if overall_quality >= 0.8:
        quality_assessment = "EXCELLENT"
    elif overall_quality >= 0.6:
        quality_assessment = "GOOD"
    elif overall_quality >= 0.4:
        quality_assessment = "MODERATE"
    else:
        quality_assessment = "POOR"

    print(f"\nOverall Quality Assessment: {quality_assessment}")

    return stats


def generate_validation_report(output_file: str = "validation_report.json"):
    """Generate comprehensive validation report"""
    print(f"\n=== GENERATING VALIDATION REPORT ===\n")

    report = {
        "validation_metadata": {
            "timestamp": str(time.time()),
            "version": "Phase1_Enhanced",
            "test_suite_version": "1.0"
        },
        "test_results": {},
        "performance_benchmarks": {},
        "scientific_validation": {},
        "recommendations": []
    }

    try:
        # Run tests and collect results
        print("Collecting test results...")
        unit_test_success = run_comprehensive_tests()
        report["test_results"]["unit_tests_passed"] = unit_test_success

        print("Collecting performance data...")
        speed_results, memory_results = run_performance_benchmarks()
        report["performance_benchmarks"]["speed"] = speed_results
        report["performance_benchmarks"]["memory"] = memory_results

        print("Collecting validation statistics...")
        validation_stats = run_scientific_validation()
        report["scientific_validation"] = validation_stats

        # Generate recommendations
        recommendations = []

        if not unit_test_success:
            recommendations.append("CRITICAL: Unit tests failed - code requires debugging before production use")

        if speed_results["avg_time_per_poem"] > 5.0:
            recommendations.append("Performance issue: Analysis taking >5s per poem - optimization needed")

        if validation_stats["quality_scores"]["mean"] < 0.6:
            recommendations.append("Quality issue: Average quality score <0.6 - algorithm tuning needed")

        if validation_stats["successful_analyses"] / validation_stats["total_poems"] < 0.8:
            recommendations.append("Reliability issue: <80% success rate - error handling improvement needed")

        if not recommendations:
            recommendations.append("All tests passed - system ready for scientific use with noted limitations")

        report["recommendations"] = recommendations

        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"Validation report saved to: {output_file}")

        # Print summary
        print(f"\n=== VALIDATION SUMMARY ===")
        print(f"Unit tests: {'PASSED' if unit_test_success else 'FAILED'}")
        print(f"Average analysis time: {speed_results['avg_time_per_poem']:.3f}s")
        print(f"Quality score: {validation_stats['quality_scores']['mean']:.3f}")
        print(f"Success rate: {validation_stats['successful_analyses'] / validation_stats['total_poems'] * 100:.1f}%")

        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")

        return report

    except Exception as e:
        print(f"Validation report generation failed: {e}")
        logger.error(f"Validation report generation failed: {e}")
        return None


def debug_analysis_failures(poem: str):
    """Debug analysis failures step by step"""
    print(f"=== DEBUGGING ANALYSIS FOR: {poem[:50]}... ===\n")

    try:
        # Test phonetic analysis
        print("1. Testing phonetic analysis...")
        from phonetic_utils import PersianPhonology
        phonology = PersianPhonology()

        phonetic, confidence = phonology.enhanced_phonetic_transcription(poem)
        print(f"   Phonetic: {phonetic}")
        print(f"   Confidence: {confidence:.3f}")

        # Test syllable boundaries
        print("\n2. Testing syllable boundaries...")
        boundaries = phonology.find_syllable_boundaries_advanced(phonetic)
        print(f"   Boundaries: {boundaries}")
        print(f"   Syllable count: {len(boundaries) - 1}")

        # Test meter analysis
        print("\n3. Testing meter analysis...")
        from enhanced_tajik_analyzer import AruzMeterAnalyzer
        meter_analyzer = AruzMeterAnalyzer()

        lines = [line.strip() for line in poem.split('\n') if line.strip()]
        for i, line in enumerate(lines):
            print(f"   Line {i + 1}: {line}")
            analysis = meter_analyzer.analyze_meter(line)
            print(f"     Meter: {analysis.identified_meter}")
            print(f"     Pattern: {analysis.pattern_match}")
            print(f"     Confidence: {analysis.confidence.value}")

        # Test rhyme analysis
        print("\n4. Testing rhyme analysis...")
        from enhanced_tajik_analyzer import AdvancedRhymeAnalyzer
        rhyme_analyzer = AdvancedRhymeAnalyzer()

        for i, line in enumerate(lines):
            rhyme_analysis = rhyme_analyzer.analyze_rhyme(line)
            print(f"   Line {i + 1}:")
            print(f"     Qāfiyeh: {rhyme_analysis.qafiyeh}")
            print(f"     Radīf: {rhyme_analysis.radif}")
            print(f"     Type: {rhyme_analysis.rhyme_type}")

        # Test complete analysis
        print("\n5. Testing complete analysis...")
        from enhanced_tajik_analyzer import EnhancedTajikPoemAnalyzer
        analyzer = EnhancedTajikPoemAnalyzer()

        result = analyzer.analyze_poem_enhanced(poem)
        print(f"   Analysis completed successfully")
        print(f"   Quality score: {result['validation']['quality_score']:.3f}")
        print(f"   Reliability: {result['validation']['reliability_level']}")

        if result['validation']['warnings']:
            print(f"   Warnings: {result['validation']['warnings']}")

    except Exception as e:
        print(f"DEBUG: Analysis failed at step: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main testing interface"""
    parser = argparse.ArgumentParser(description="Tajik Poetry Analyzer Testing Suite")
    parser.add_argument("--run-all", action="store_true", help="Run all tests and generate report")
    parser.add_argument("--test", choices=["phonetic", "meter", "rhyme", "integration"],
                        help="Run specific test category")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--validate", type=str, help="Run scientific validation on results file")
    parser.add_argument("--debug", type=str, help="Debug analysis failures for given poem")
    parser.add_argument("--report", type=str, default="validation_report.json",
                        help="Output file for validation report")

    args = parser.parse_args()

    if args.run_all:
        generate_validation_report(args.report)

    elif args.test:
        if args.test == "phonetic":
            suite = unittest.TestLoader().loadTestsFromTestCase(PhoneticAnalysisTests)
        elif args.test == "meter":
            suite = unittest.TestLoader().loadTestsFromTestCase(AruzMeterTests)
        elif args.test == "rhyme":
            suite = unittest.TestLoader().loadTestsFromTestCase(RhymeAnalysisTests)
        elif args.test == "integration":
            suite = unittest.TestLoader().loadTestsFromTestCase(IntegrationTests)

        unittest.TextTestRunner(verbosity=2).run(suite)

    elif args.benchmark:
        run_performance_benchmarks()

    elif args.validate:
        run_scientific_validation(args.validate)

    elif args.debug:
        debug_analysis_failures(args.debug)

    else:
        print("No action specified. Use --help for options.")
        print("\nQuick start:")
        print("  python test_validation_suite.py --run-all")
        print("  python test_validation_suite.py --test integration")
        print("  python test_validation_suite.py --debug 'Your poem text here'")


if __name__ == "__main__":
    main()
