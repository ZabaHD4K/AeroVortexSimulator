#pragma once
#include <string>
#include <vector>

class App; // forward declaration

class Gui {
public:
    void init(const std::string& modelsPath);
    void render(App& app);

    bool wantsLoadModel() const { return loadRequested; }
    void clearLoadRequest() { loadRequested = false; }

    bool wantsLoadFromLibrary() const { return libraryLoadRequested; }
    void clearLibraryRequest() { libraryLoadRequested = false; }
    const std::string& getLibrarySelection() const { return selectedLibraryPath; }

    bool wantsLBMReset() const { return lbmResetRequested; }
    void clearLBMReset() { lbmResetRequested = false; }

    bool wantsSimInit() const { return simInitRequested; }
    void clearSimInit() { simInitRequested = false; }

    bool wantsSimReset() const { return simResetRequested; }
    void clearSimReset() { simResetRequested = false; }

    bool wantsWindStart() const { return windStartRequested; }
    void clearWindStart() { windStartRequested = false; }

    // Test model generation
    bool wantsTestSphere() const { return testSphereRequested; }
    void clearTestSphere() { testSphereRequested = false; }
    bool wantsTestCylinder() const { return testCylinderRequested; }
    void clearTestCylinder() { testCylinderRequested = false; }
    bool wantsTestNACA() const { return testNACARequested; }
    void clearTestNACA() { testNACARequested = false; }

    bool wantsWindDirChange() const { return windDirChangeRequested; }
    void clearWindDirChange() { windDirChangeRequested = false; }

    bool wantsScreenshot() const { return screenshotRequested; }
    void clearScreenshot() { screenshotRequested = false; }

    bool wantsExportVTK() const { return exportVTKRequested; }
    void clearExportVTK() { exportVTKRequested = false; }

    bool wantsExportCSV() const { return exportCSVRequested; }
    void clearExportCSV() { exportCSVRequested = false; }

    bool wantsExportReport() const { return exportReportRequested; }
    void clearExportReport() { exportReportRequested = false; }

    bool wantsExportFlowCSV() const { return exportFlowCSVRequested; }
    void clearExportFlowCSV() { exportFlowCSVRequested = false; }

    void refreshModelLibrary();

private:
    bool loadRequested = false;
    bool libraryLoadRequested = false;
    bool lbmResetRequested = false;
    bool simInitRequested = false;
    bool simResetRequested = false;
    bool windStartRequested = false;
    bool windDirChangeRequested = false;
    bool screenshotRequested = false;
    bool exportVTKRequested = false;
    bool exportCSVRequested = false;
    bool exportReportRequested = false;
    bool exportFlowCSVRequested = false;
    bool testSphereRequested = false;
    bool testCylinderRequested = false;
    bool testNACARequested = false;
    std::string selectedLibraryPath;

    std::string modelsDir;
    struct ModelEntry {
        std::string name;
        std::string path;
        std::string extension;
        float sizeMB;
    };
    std::vector<ModelEntry> libraryModels;
};

std::string openFileDialog();
