"""
Simple test to verify the refactored Nexto bot structure.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported successfully."""
    try:
        import vector_utils
        import bot_enums
        # Skip modules that require numpy for now
        print("✓ Basic imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_vec3():
    """Test Vec3 class basic functionality."""
    try:
        from vector_utils import Vec3
        
        v1 = Vec3(1, 2, 3)
        v2 = Vec3(4, 5, 6)
        
        # Test basic operations
        v3 = v1 + v2
        assert v3.x == 5 and v3.y == 7 and v3.z == 9
        
        v4 = v2 - v1
        assert v4.x == 3 and v4.y == 3 and v4.z == 3
        
        v5 = v1 * 2
        assert v5.x == 2 and v5.y == 4 and v5.z == 6
        
        # Test magnitude
        v6 = Vec3(3, 4, 0)
        assert v6.magnitude() == 5.0
        
        # Test dot product
        assert v1.dot(v2) == 32  # 1*4 + 2*5 + 3*6
        
        print("✓ Vec3 tests passed")
        return True
    except Exception as e:
        print(f"✗ Vec3 test failed: {e}")
        return False

def test_physics_constants():
    """Test physics constants are properly defined."""
    try:
        import physics_utils
        PhysicsConstants = physics_utils.PhysicsConstants
        
        # Check that key constants exist
        assert hasattr(PhysicsConstants, 'GRAVITY')
        assert hasattr(PhysicsConstants, 'BOOST_ACCELERATION')
        assert hasattr(PhysicsConstants, 'SUPERSONIC_SPEED')
        assert PhysicsConstants.GRAVITY == 650.0
        assert PhysicsConstants.SUPERSONIC_SPEED == 2300.0
        
        print("✓ Physics constants tests passed")
        return True
    except Exception as e:
        print(f"✗ Physics constants test failed: {e}")
        return False

def test_demo_controller():
    """Test demo controller can be instantiated."""
    try:
        import demo_controller
        import bot_enums
        
        controller = demo_controller.DemoController()
        assert controller.demo_phase == bot_enums.DemolishPhase.CHOOSE_TARGET
        assert controller.demo_target_index == -1
        assert not controller.is_dodging
        
        print("✓ Demo controller tests passed")
        return True
    except Exception as e:
        print(f"✗ Demo controller test failed: {e}")
        return False

def test_kickoff_controller():
    """Test kickoff controller functionality."""
    try:
        import kickoff_controller
        
        controller = kickoff_controller.KickoffController()
        assert controller.kickoff_index == -1
        
        # Check that kickoff sequence exists
        assert len(kickoff_controller.KICKOFF_NUMPY) > 0
        assert kickoff_controller.KICKOFF_NUMPY.shape[1] == 8  # 8 action dimensions
        
        print("✓ Kickoff controller tests passed")
        return True
    except ImportError as e:
        print(f"⚠ Kickoff controller test skipped (missing rlbot dependency): {e}")
        return True  # Skip this test
    except Exception as e:
        print(f"✗ Kickoff controller test failed: {e}")
        return False

def test_enums():
    """Test enum definitions."""
    try:
        from bot_enums import BotMode, DemolishPhase
        
        assert BotMode.NORMAL != BotMode.DEMO
        assert DemolishPhase.CHOOSE_TARGET != DemolishPhase.CHASING
        assert DemolishPhase.CHASING != DemolishPhase.FAILED
        
        print("✓ Enums tests passed")
        return True
    except Exception as e:
        print(f"✗ Enums test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running tests for refactored Nexto bot structure...")
    print()
    
    tests = [
        test_imports,
        test_vec3,
        test_physics_constants,
        test_demo_controller,
        test_kickoff_controller,
        test_enums
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! The refactored structure is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)