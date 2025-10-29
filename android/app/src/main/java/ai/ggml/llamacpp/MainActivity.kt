package ai.ggml.llamacpp


import ai.ggml.llamacpp.ui.theme.LlamaAndroidTheme
import android.annotation.SuppressLint
import android.app.ActivityManager
import android.app.DownloadManager
import android.content.Context
import android.icu.util.TimeUnit
import android.os.Bundle
import android.os.HardwarePropertiesManager
import android.os.PowerManager
import android.os.StrictMode
import android.os.StrictMode.VmPolicy
import android.text.format.Formatter
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.systemBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.DeviceThermostat
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.PrimaryTabRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.core.content.getSystemService
import androidx.core.net.toUri
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import kotlinx.coroutines.launch
import java.io.File
import kotlinx.coroutines.delay
import android.util.Log

private fun getMemoryUsageString(context: Context): String {
    val activityManager = context.getSystemService<ActivityManager>()
    val memoryInfo = ActivityManager.MemoryInfo()
    activityManager?.getMemoryInfo(memoryInfo)

    val usedMem = memoryInfo.totalMem - memoryInfo.availMem
    val usedMemGiB = usedMem / 1024.0 / 1024.0 / 1024.0
    val totalMemGiB = memoryInfo.totalMem / 1024.0 / 1024.0 / 1024.0

    return "${"%.1f".format(usedMemGiB)} / ${"%.1f".format(totalMemGiB)} GiB"
}

fun thermalStatusToString(status: Int): String {
    return when (status) {
        PowerManager.THERMAL_STATUS_NONE -> "None"
        PowerManager.THERMAL_STATUS_LIGHT -> "Light"
        PowerManager.THERMAL_STATUS_MODERATE -> "Moderate"
        PowerManager.THERMAL_STATUS_SEVERE -> "Severe"
        PowerManager.THERMAL_STATUS_CRITICAL -> "Critical"
        PowerManager.THERMAL_STATUS_EMERGENCY -> "Emergency"
        PowerManager.THERMAL_STATUS_SHUTDOWN -> "Shutdown"
        else -> "Unknown ($status)"
    }
}

private fun getThermalStatusString(context: Context): String {
    val powerManager = context.getSystemService<PowerManager>()
    val thermalStatus: Int = powerManager!!.currentThermalStatus
    return thermalStatusToString(thermalStatus)
}

var tag = "llama.app"

class MainActivity(
    downloadManager: DownloadManager? = null,
): ComponentActivity() {



    private val downloadManager by lazy { downloadManager ?: getSystemService<DownloadManager>()!! }

    private val viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        StrictMode.setVmPolicy(
            VmPolicy.Builder(StrictMode.getVmPolicy())
                .detectLeakedClosableObjects()
                .build()
        )

        viewModel.log("Downloads directory: ${getExternalFilesDir(null)}")

        val extFilesDir = getExternalFilesDir(null)

        val models = listOf(
            Downloadable(
                "Qwen3 1.7B (Q4_0, 1.06 GiB)",
                "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_0.gguf?download=true".toUri(),
                File(extFilesDir, "Qwen3-1.7B-Q4_0.gguf"),
            ),
            Downloadable(
                "Qwen3 0.6B (Q8_0, 639 MiB)",
                "https://huggingface.co/prithivMLmods/Qwen3-0.6B-GGUF/resolve/main/Qwen3_0.6B.Q8_0.gguf?download=true".toUri(),
                File(extFilesDir, "Qwen3_0.6B.Q8_0.gguf"),
            ),
            Downloadable(
                "Gemma 3 1B (Q4_0, 722 MiB)",
                "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_0.gguf?download=true".toUri(),
                File(extFilesDir, "gemma-3-1b-it-Q4_0.gguf"),
            ),
            Downloadable(
                "Gemma 3 4B (Q8_0, 4.13 GiB)",
                "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q8_0.gguf?download=true".toUri(),
                File(extFilesDir, "gemma-3-4b-it-Q8_0.gguf"),
            ),
            Downloadable(
                "Phi-2 7B (Q4_0, 1.6 GiB)",
                "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q4_0.gguf?download=true".toUri(),
                File(extFilesDir, "phi-2-q4_0.gguf"),
            ),
            Downloadable(
                "TinyLlama 1.1B (f16, 2.2 GiB)",
                "https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b/ggml-model-f16.gguf?download=true".toUri(),
                File(extFilesDir, "tinyllama-1.1-f16.gguf"),
            ),
            Downloadable(
                "Phi 2 DPO (Q3_K_M, 1.48 GiB)",
                "https://huggingface.co/TheBloke/phi-2-dpo-GGUF/resolve/main/phi-2-dpo.Q3_K_M.gguf?download=true".toUri(),
                File(extFilesDir, "phi-2-dpo.Q3_K_M.gguf")
            ),
        )

        setContent {
            LlamaAndroidTheme {
                // A surface container using the 'background' color from the theme

                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainCompose(
                        viewModel,
                        downloadManager,
                        models,
                    )
                }

            }
        }
    }
}

enum class Destination(
    val route: String,
    val label: String,
    val icon: ImageVector,
    val contentDescription: String
) {
    CHAT("chat", "Chat", Icons.AutoMirrored.Filled.Send, "Chat"),
    BENCHMARK("benchmark", "Benchmark", Icons.AutoMirrored.Filled.Send, "Benchmark"),
    FINE_TUNING("finetune", "Fine-tune", Icons.AutoMirrored.Filled.Send, "Fine-tune")
}

@Composable
fun ChatScreen(
    viewModel: MainViewModel
) {
    Column (
        modifier = Modifier.imePadding()
    ) {
        val scrollState = rememberLazyListState()
        val coroutineScope = rememberCoroutineScope()

        Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
            LazyColumn(state = scrollState) {
                items(viewModel.messages) {
                    Text(it)
                }
                coroutineScope.launch {
                    scrollState.animateScrollToItem(viewModel.messages.lastIndex)
                }
            }
        }
        OutlinedTextField(
            value = viewModel.message,
            onValueChange = { viewModel.updateMessage(it) },
            label = { Text("Prompt") },
            modifier = Modifier.fillMaxWidth(),
        )
        Box (
            modifier = Modifier.fillMaxWidth(),
            contentAlignment = Alignment.TopEnd
        ) {
            IconButton(
                onClick = { viewModel.send() },
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.Send,
                    contentDescription = "Send",
                )
            }
        }
    }
}



@Composable
fun BenchmarkScreen(viewModel: MainViewModel) {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Button({ viewModel.bench(8, 4, 1) }) { Text("Bench") }
    }
}

@Composable
fun FineTuningScreen(
    viewModel: MainViewModel,
) {
    Text("Fine tuning")
}

@Composable
fun ModelsBottomSheet(
    viewModel: MainViewModel,
    dm: DownloadManager?,
    models: List<Downloadable>
) {
    Column (
        modifier = Modifier.fillMaxSize()
    ) {
        Text("Choose model")
        Text("WIP: Don't close this view during download!")
//        Button (
//            onClick = {}
//        ) {
//            Text("Load model from file...")
//        }
        for (model in models) {
            Downloadable.Button(viewModel, dm, model)
        }
    }
}

@Composable
fun AppNavHost(
    navController: NavHostController,
    startDestination: Destination,
    viewModel: MainViewModel
) {
    NavHost(
        navController,
        startDestination = startDestination.route
    ) {
        Destination.entries.forEach { destination ->
            composable(destination.route) {
                when (destination) {
                    Destination.CHAT -> ChatScreen(viewModel)
                    Destination.BENCHMARK -> BenchmarkScreen(viewModel)
                    Destination.FINE_TUNING -> FineTuningScreen(viewModel)
                }
            }
        }
    }
}

@Composable
fun MemoryUsageText() {
    val context = LocalContext.current
    var memoryText by remember { mutableStateOf("? / ? GiB") }

    LaunchedEffect(Unit) {
        while (true) {
            try {
                memoryText = getMemoryUsageString(context)
            } catch (e: Exception) {
                Log.e(tag, "Error fetching temperature: ${e.message}");
            }
            delay(2000)
        }
    }

    Text(
        text = memoryText
    )
}

@Composable
fun TemperatureText() {
    val context = LocalContext.current
    var temperatureText by remember { mutableStateOf("?") }

    LaunchedEffect(Unit) {
        while (true) {
            try {
                temperatureText = getThermalStatusString(context)
            } catch (e: Exception) {
                Log.e(tag, "Error fetching temperature: ${e.message}");
            }
            delay(2000)
        }
    }

    Row (
        verticalAlignment = Alignment.CenterVertically
    )
    {
        Icon(
            imageVector = Icons.Filled.DeviceThermostat,
            contentDescription = "Temperature",
        )
        Text(
            text = temperatureText
        )
    }
}



@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainCompose(
    viewModel: MainViewModel,
    dm: DownloadManager?,
    models: List<Downloadable>
) {
    val navController = rememberNavController()
    val startDestination = Destination.CHAT
    var selectedDestination by rememberSaveable { mutableIntStateOf(startDestination.ordinal) }

    var showBottomSheet by remember { mutableStateOf(false) }
    val sheetState = rememberModalBottomSheetState(
        skipPartiallyExpanded = false,
    )

    Column (
        modifier = Modifier.windowInsetsPadding(WindowInsets.systemBars)
    ) {
        Box (
            contentAlignment = Alignment.Center
        ) {
            Box (
                modifier = Modifier.fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                Button(
                    onClick = { showBottomSheet = true }
                ) {
                    Text(viewModel.currentModelName)
                    Icon(
                        imageVector = Icons.Filled.ArrowDropDown,
                        contentDescription = "Send",
                    )
                }
            }
            Row (
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                MemoryUsageText()
                Box (
                    modifier = Modifier.weight(1f),
                    contentAlignment = Alignment.Center
                ) {
                }
                TemperatureText()
            }
        }
        PrimaryTabRow(selectedTabIndex = selectedDestination) {
            Destination.entries.forEachIndexed { index, destination ->
                Tab(
                    selected = selectedDestination == index,
                    onClick = {
                        navController.navigate(route = destination.route)
                        selectedDestination = index
                    },
                    text = {
                        Text(
                            text = destination.label,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                )
            }
        }
        AppNavHost(navController, startDestination, viewModel)

        if (showBottomSheet) {
            ModalBottomSheet(
                modifier = Modifier.fillMaxHeight(),
                sheetState = sheetState,
                onDismissRequest = { showBottomSheet = false }
            ) {
                ModelsBottomSheet(viewModel, dm, models)
            }
        }
    }
}

@SuppressLint("ViewModelConstructorInComposable")
@Preview
@Composable
fun PreviewMainCompose() {
    val viewModel = MainViewModel()

    val models = listOf(
        Downloadable(
            "Phi-2 7B (Q4_0, 1.6 GiB)",
            "".toUri(),
            File(""),
        ),
        Downloadable(
            "TinyLlama 1.1B (f16, 2.2 GiB)",
            "".toUri(),
            File(""),
        ),
        Downloadable(
            "Phi 2 DPO (Q3_K_M, 1.48 GiB)",
            "".toUri(),
            File("")
        ),
    )
    MainCompose(
        viewModel,
        null,
        models
    )
}

@SuppressLint("ViewModelConstructorInComposable")
@Preview
@Composable
fun PreviewModelsBottomSheet() {
    val viewModel = MainViewModel()

    val models = listOf(
        Downloadable(
            "Phi-2 7B (Q4_0, 1.6 GiB)",
            "".toUri(),
            File(""),
        ),
        Downloadable(
            "TinyLlama 1.1B (f16, 2.2 GiB)",
            "".toUri(),
            File(""),
        ),
        Downloadable(
            "Phi 2 DPO (Q3_K_M, 1.48 GiB)",
            "".toUri(),
            File("")
        ),
    )
    ModelsBottomSheet(viewModel, null, models)
}
