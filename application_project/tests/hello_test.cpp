#include <gtest/gtest.h>
#include <stdio.h>
#pragma message("Compiling " __FILE__)

//#include "code.h"

//#include <torch/torch.h>

//#include "../functions/common.h"

/*
* The directory for tests is:
* out/build/x64-debug/application_project/tests
* This thing just refuses to build with libtorch for some reason
* TODO: get this thing working
*/

TEST(HelloTest, BasicAssertion)
{
	//hello_world();
	EXPECT_STRNE("hello", "world");
	EXPECT_EQ(7 * 6, 42);
}

int main(int argc, char *argv[])
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

