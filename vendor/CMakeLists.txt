function(download_file url filename hash_type hash)

    if(NOT EXISTS ${filename})
        file(DOWNLOAD ${url} ${filename}
            TIMEOUT 60  # seconds
            EXPECTED_HASH ${hash_type}=${hash}
            TLS_VERIFY ON)
    endif()

endfunction(download_file)

download_file(https://www.stata.com/plugins/stplugin.h ${CMAKE_CURRENT_LIST_DIR}/stplugin.h MD5 4cdc8bd8323266b4c79c3095b47d8685)
download_file(https://www.stata.com/plugins/stplugin.c ${CMAKE_CURRENT_LIST_DIR}/stplugin.cpp MD5 b23326de8394154f93cc232b33fe2700)
