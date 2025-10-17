package ai.ggml.llamacpp


import ai.ggml.llamacpp.ui.theme.LlamaAndroidTheme
import android.annotation.SuppressLint
import android.app.ActivityManager
import android.app.DownloadManager
import android.os.Bundle
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

class MainActivity(
    activityManager: ActivityManager? = null,
    downloadManager: DownloadManager? = null,
): ComponentActivity() {

    private val activityManager by lazy { activityManager ?: getSystemService<ActivityManager>()!! }
    private val downloadManager by lazy { downloadManager ?: getSystemService<DownloadManager>()!! }

    private val viewModel: MainViewModel by viewModels()

    // Get a MemoryInfo object for the device's current memory status.
    private fun availableMemory(): ActivityManager.MemoryInfo {
        return ActivityManager.MemoryInfo().also { memoryInfo ->
            activityManager.getMemoryInfo(memoryInfo)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        StrictMode.setVmPolicy(
            VmPolicy.Builder(StrictMode.getVmPolicy())
                .detectLeakedClosableObjects()
                .build()
        )

        val free = Formatter.formatFileSize(this, availableMemory().availMem)
        val total = Formatter.formatFileSize(this, availableMemory().totalMem)

        viewModel.log("Current memory: $free / $total")
        viewModel.log("Downloads directory: ${getExternalFilesDir(null)}")

        val extFilesDir = getExternalFilesDir(null)

        val models = listOf(
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
    MODELS("models", "Models", Icons.AutoMirrored.Filled.Send, "Models")
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
fun ModelsScreen(
    viewModel: MainViewModel,
    dm: DownloadManager?,
    models: List<Downloadable>
) {
    Column (
        modifier = Modifier.fillMaxSize()
    ) {
        for (model in models) {
            Downloadable.Button(viewModel, dm, model)
        }
    }
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
        for (model in models) {
            Downloadable.Button(viewModel, dm, model)
        }
    }
}

@Composable
fun AppNavHost(
    navController: NavHostController,
    startDestination: Destination,
    viewModel: MainViewModel,
    dm: DownloadManager?,
    models: List<Downloadable>
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
                    Destination.MODELS -> ModelsScreen(viewModel, dm, models)
                }
            }
        }
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
                    Text("No Model")
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
                Text("14 / 15 GB")
                Box (
                    modifier = Modifier.weight(1f),
                    contentAlignment = Alignment.Center
                ) {
                }
                Text("35 C")
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
        AppNavHost(navController, startDestination, viewModel, dm, models)

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
