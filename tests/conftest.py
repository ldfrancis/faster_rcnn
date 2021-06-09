import pytest

from tests.mocks import MockDataset, MockFeatureMap, mock_config


@pytest.fixture
def cfg():
    return mock_config


@pytest.fixture
def dataset():
    return MockDataset()


@pytest.fixture
def image():
    return MockDataset.get_image()


@pytest.fixture
def image_featmap():
    return MockFeatureMap.create_featmap()


@pytest.fixture
def pooled_roi_7x7():
    return MockFeatureMap.create_7x7_pooled_roi()


@pytest.fixture
def processed_roi():
    return MockFeatureMap.create_processed_roi()
