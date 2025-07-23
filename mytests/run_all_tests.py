#!/usr/bin/env python3

"""
Comprehensive test runner for all SequenceHookOutputProcessor tests.

This script runs all test suites and provides a summary of results.
"""

import sys
import time
import traceback
from typing import List, Callable, Tuple

# Import all test modules
try:
    from test_sequence_hook_processor import run_tests as run_processor_tests
    from test_processor_factory import run_factory_tests
    from test_edge_cases import run_edge_case_tests
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the mytests directory")
    sys.exit(1)


class TestRunner:
    """Manages running all test suites with reporting."""
    
    def __init__(self):
        self.results: List[Tuple[str, bool, float, str]] = []
    
    def run_test_suite(self, name: str, test_func: Callable) -> bool:
        """Run a single test suite and record results."""
        print(f"\n{'='*60}")
        print(f"🧪 Running {name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        success = False
        error_msg = ""
        
        try:
            test_func()
            success = True
            print(f"✅ {name} completed successfully")
        except Exception as e:
            success = False
            error_msg = str(e)
            print(f"❌ {name} failed: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")
        
        duration = time.time() - start_time
        self.results.append((name, success, duration, error_msg))
        
        return success
    
    def run_all_tests(self) -> bool:
        """Run all test suites."""
        print("🚀 Starting SequenceHookOutputProcessor Test Suite")
        print(f"{'='*60}")
        
        test_suites = [
            ("Core Processor Tests", run_processor_tests),
            ("Factory Method Tests", run_factory_tests),
            ("Edge Case Tests", run_edge_case_tests),
        ]
        
        all_passed = True
        
        for name, test_func in test_suites:
            success = self.run_test_suite(name, test_func)
            if not success:
                all_passed = False
        
        self.print_summary()
        return all_passed
    
    def print_summary(self):
        """Print test results summary."""
        print(f"\n{'='*60}")
        print("📊 TEST SUMMARY")
        print(f"{'='*60}")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for _, success, _, _ in self.results if success)
        failed_tests = total_tests - passed_tests
        total_time = sum(duration for _, _, duration, _ in self.results)
        
        print(f"Total Test Suites: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Total Time: {total_time:.2f}s")
        print()
        
        # Detailed results
        for name, success, duration, error_msg in self.results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status:<8} {name:<25} ({duration:.2f}s)")
            if not success and error_msg:
                print(f"         Error: {error_msg}")
        
        print(f"\n{'='*60}")
        if failed_tests == 0:
            print("🎉 ALL TESTS PASSED! 🎉")
        else:
            print(f"⚠️  {failed_tests} TEST SUITE(S) FAILED")
        print(f"{'='*60}")


def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality."""
    print("🔥 Running quick smoke test...")
    
    try:
        from vllm.config import SpecialKwargs
        from vllm.engine.output_processor.sequence_hook import SequenceHookOutputProcessor
        
        # Test basic imports and construction
        special_kwargs = SpecialKwargs(sync_token_id=123, promise_token_id=456)
        assert special_kwargs.sync_token_id == 123
        assert special_kwargs.promise_token_id == 456
        
        print("✅ Smoke test passed - basic imports and construction work")
        return True
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False


def run_performance_test():
    """Run a simple performance test to ensure no major overhead."""
    print("⚡ Running performance test...")
    
    try:
        # This would be a more comprehensive performance test
        # For now, just test that we can create processors quickly
        start_time = time.time()
        
        from vllm.config import SpecialKwargs
        special_kwargs = SpecialKwargs(sync_token_id=123, promise_token_id=456)
        
        # Create multiple instances to test overhead
        for i in range(100):
            _ = SpecialKwargs(sync_token_id=i, promise_token_id=i+1000)
        
        duration = time.time() - start_time
        
        if duration < 1.0:  # Should be very fast
            print(f"✅ Performance test passed - 100 creations took {duration:.4f}s")
            return True
        else:
            print(f"⚠️  Performance test slow - took {duration:.4f}s")
            return False
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def main():
    """Main test runner entry point."""
    print("🧪 SequenceHookOutputProcessor Test Suite")
    print("=" * 60)
    
    # Run smoke test first
    if not run_quick_smoke_test():
        print("❌ Smoke test failed - skipping remaining tests")
        sys.exit(1)
    
    # Run performance test
    run_performance_test()
    
    # Run all comprehensive tests
    runner = TestRunner()
    all_passed = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


def run_specific_test(test_name: str):
    """Run a specific test suite by name."""
    test_map = {
        "processor": run_processor_tests,
        "factory": run_factory_tests,
        "edge": run_edge_case_tests,
    }
    
    if test_name not in test_map:
        print(f"❌ Unknown test: {test_name}")
        print(f"Available tests: {list(test_map.keys())}")
        sys.exit(1)
    
    print(f"🧪 Running specific test: {test_name}")
    try:
        test_map[test_name]()
        print(f"✅ {test_name} test completed successfully")
    except Exception as e:
        print(f"❌ {test_name} test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check for specific test argument
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        run_specific_test(test_name)
    else:
        main() 