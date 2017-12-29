package main

import "testing"

func Test(t *testing.T) {
	// Arrange

	// Act
	if sigmoid(0) != 0.5 {
		t.Error("Unexpected result")
	}
	// Assert
}
